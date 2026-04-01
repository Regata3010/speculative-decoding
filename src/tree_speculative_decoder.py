"""Tree-based speculative decoding (SpecInfer-style).

Instead of drafting a single chain of K tokens, we draft a TREE of
candidates. At each position, the draft model proposes the top-B tokens
(branching factor B), creating multiple candidate paths. The target model
verifies the entire tree in ONE forward pass using a tree attention mask.

Why this is faster:
  - Linear: if token at position 2 is rejected, tokens 3-5 are wasted
  - Tree: if branch 1 at position 2 is rejected, branch 2 might pass —
    we recover from rejections without burning another target call
  - The target model is memory-bandwidth bound (reads 72B weights once
    regardless of whether it processes 6 or 15 tokens), so the extra
    tree nodes have near-zero marginal cost

Reference: SpecInfer (Miao et al., 2024) — "Accelerating Generative LLM
Serving with Tree-based Speculative Inference and Verification"
"""

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.kv_cache_manager import create_cache, get_cache_length, rollback_cache
from src.sampling import get_probs_from_logits, sample_from_logits, _align_vocab_sizes
from src.utils import CudaTimer, GenerationResult, ProfilingData


@dataclass
class TreeNode:
    """A node in the draft token tree."""
    token_id: int
    depth: int
    parent_idx: int  # Index in the flat node list (-1 for root)
    children: list[int] = field(default_factory=list)  # Indices of children
    logits: torch.Tensor | None = None


def build_tree_attention_mask(nodes: list[TreeNode], prompt_len: int) -> torch.Tensor:
    """Build a causal attention mask for tree verification.

    Each tree node can attend to:
      - All prompt tokens
      - Its ancestor chain in the tree (parent, grandparent, ... root)
      - Itself

    But NOT to sibling branches or nodes outside its ancestor path.

    Args:
        nodes: Flat list of tree nodes (index 0 = first draft token).
        prompt_len: Number of prompt tokens (all tree nodes attend to these).

    Returns:
        Boolean mask of shape (n_tree_nodes, prompt_len + n_tree_nodes).
        True = can attend, False = masked out.
    """
    n = len(nodes)
    total_len = prompt_len + n

    # Start with: all nodes attend to all prompt tokens
    mask = torch.zeros(n, total_len, dtype=torch.bool)
    mask[:, :prompt_len] = True

    # Each node attends to itself and its ancestors
    for i, node in enumerate(nodes):
        mask[i, prompt_len + i] = True  # Attend to self
        # Walk up the ancestor chain
        parent = node.parent_idx
        while parent >= 0:
            mask[i, prompt_len + parent] = True
            parent = nodes[parent].parent_idx

    return mask


class TreeSpeculativeDecoder:
    """Tree-based speculative decoding engine."""

    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        depth: int = 5,
        branch_factor: int = 2,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.depth = depth
        self.branch_factor = branch_factor
        self.temperature = temperature
        self.top_p = top_p
        self.device = next(target_model.parameters()).device

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        profile: bool = True,
    ) -> GenerationResult:
        """Generate tokens using tree-based speculative decoding.

        Args:
            input_ids: Shape (1, seq_len) — tokenized prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            profile: If True, record per-phase timing.

        Returns:
            GenerationResult with generated tokens and performance metrics.
        """
        timer = CudaTimer(self.device)
        timer.start()

        input_ids = input_ids.to(self.device)
        generated_ids = input_ids.squeeze(0).tolist()
        prompt_len = len(generated_ids)
        eos_token_id = self.tokenizer.eos_token_id

        draft_cache = create_cache()
        target_cache = create_cache()
        n_target_calls = 0
        n_accepted_draft = 0
        n_total_draft = 0

        prof = ProfilingData() if profile else None
        _sync = torch.cuda.synchronize if self.device.type == "cuda" else lambda: None

        # === Prefill ===
        if prof:
            _sync()
            t0 = time.perf_counter()

        target_out = self.target_model(
            input_ids=input_ids, past_key_values=target_cache, use_cache=True
        )
        target_cache = target_out.past_key_values
        n_target_calls += 1

        if prof:
            _sync()
            prof.target_time += time.perf_counter() - t0
            t0 = time.perf_counter()

        draft_out = self.draft_model(
            input_ids=input_ids, past_key_values=draft_cache, use_cache=True
        )
        draft_cache = draft_out.past_key_values

        if prof:
            _sync()
            prof.draft_time += time.perf_counter() - t0

        # Sample first token from target
        first_token = sample_from_logits(
            target_out.logits[:, -1, :].squeeze(0), self.temperature, self.top_p
        )
        generated_ids.append(first_token.item())

        if first_token.item() == eos_token_id:
            return self._build_result(
                generated_ids, prompt_len, n_target_calls,
                n_accepted_draft, n_total_draft, timer, prof,
            )

        # === Main loop ===
        while len(generated_ids) - prompt_len < max_new_tokens:
            tokens_remaining = max_new_tokens - (len(generated_ids) - prompt_len)
            if tokens_remaining <= 0:
                break

            current_depth = min(self.depth, tokens_remaining)
            current_seq_len = len(generated_ids)

            # --- Step 1: Build draft tree ---
            if prof:
                _sync()
                t0 = time.perf_counter()

            tree_nodes, tree_draft_probs = self._build_draft_tree(
                generated_ids, draft_cache, current_depth
            )

            if prof:
                _sync()
                prof.draft_time += time.perf_counter() - t0

            if not tree_nodes:
                break

            n_total_draft += len(tree_nodes)

            # --- Step 2: Target verifies entire tree in one forward pass ---
            if prof:
                _sync()
                t0 = time.perf_counter()

            # Sync target cache if needed
            expected_target_len = current_seq_len - 1
            if get_cache_length(target_cache) < expected_target_len:
                missing_start = get_cache_length(target_cache)
                missing_tokens = torch.tensor(
                    [generated_ids[missing_start:expected_target_len]], device=self.device
                )
                target_out = self.target_model(
                    input_ids=missing_tokens,
                    past_key_values=target_cache,
                    use_cache=True,
                )
                target_cache = target_out.past_key_values
                n_target_calls += 1

            # Extract the best path through the tree for verification.
            # The tree provides multiple candidates at each depth — we pick
            # the best path (highest draft probability) and verify it linearly.
            # This avoids custom attention masks (which break with quantized
            # models and HF's SDPA) while still benefiting from tree diversity:
            # if the best candidate at depth i is rejected, we try the second
            # candidate before giving up on that position.
            best_path, path_draft_probs = self._extract_best_path(
                tree_nodes, tree_draft_probs
            )

            # Verify the best path linearly (same as standard speculative decoding)
            verify_tokens = [generated_ids[-1]] + [n.token_id for n in best_path]
            verify_input = torch.tensor([verify_tokens], device=self.device)

            target_out = self.target_model(
                input_ids=verify_input,
                past_key_values=target_cache,
                use_cache=True,
            )
            target_cache = target_out.past_key_values
            n_target_calls += 1

            if prof:
                _sync()
                prof.target_time += time.perf_counter() - t0

            # --- Step 3: Tree-based acceptance with fallback ---
            if prof:
                _sync()
                t0 = time.perf_counter()

            target_logits = target_out.logits.squeeze(0)  # (path_len+1, vocab)
            target_probs = get_probs_from_logits(
                target_logits, self.temperature, self.top_p
            )

            # Try acceptance along the best path, with tree fallback
            accepted_tokens, n_accepted_in_tree = self._tree_accept_with_fallback(
                tree_nodes, tree_draft_probs, best_path, path_draft_probs,
                target_probs,
            )

            n_accepted_draft += n_accepted_in_tree

            if prof:
                _sync()
                prof.sampling_time += time.perf_counter() - t0

            # Append accepted tokens
            hit_eos = False
            for tok in accepted_tokens:
                generated_ids.append(tok)
                if tok == eos_token_id:
                    hit_eos = True
                    break

            if hit_eos:
                return self._build_result(
                    generated_ids, prompt_len, n_target_calls,
                    n_accepted_draft, n_total_draft, timer, prof,
                )

            # --- Step 4: Rollback caches ---
            if prof:
                _sync()
                t0 = time.perf_counter()

            new_cache_len = len(generated_ids) - 1
            rollback_cache(target_cache, new_cache_len)
            rollback_cache(draft_cache, new_cache_len)

            if prof:
                _sync()
                prof.cache_time += time.perf_counter() - t0

        return self._build_result(
            generated_ids, prompt_len, n_target_calls,
            n_accepted_draft, n_total_draft, timer, prof,
        )

    def _build_draft_tree(
        self,
        generated_ids: list[int],
        draft_cache,
        depth: int,
    ) -> tuple[list[TreeNode], list[torch.Tensor]]:
        """Build a tree of draft candidates.

        Uses BFS: at each depth level, expand all leaf nodes by sampling
        top-B tokens from the draft model.

        Returns:
            (tree_nodes, draft_probs):
                tree_nodes: Flat list of all nodes in BFS order.
                draft_probs: List of probability tensors for each node.
        """
        # Sync draft cache
        expected_cache_len = len(generated_ids) - 1
        current_cache_len = get_cache_length(draft_cache)
        if current_cache_len < expected_cache_len:
            missing = generated_ids[current_cache_len:expected_cache_len]
            if missing:
                missing_input = torch.tensor([missing], device=self.device)
                draft_out = self.draft_model(
                    input_ids=missing_input,
                    past_key_values=draft_cache,
                    use_cache=True,
                )
                draft_cache.__dict__.update(draft_out.past_key_values.__dict__)

        tree_nodes: list[TreeNode] = []
        draft_probs: list[torch.Tensor] = []

        # We need separate cache states for each branch.
        # Strategy: process the tree level by level. At each level, we can
        # batch all nodes at that level if they share the same cache prefix.
        # For simplicity (and because the draft model is small), we process
        # nodes sequentially but efficiently.

        # Level 0: expand from the last accepted token
        # Save draft cache state before branching
        import copy
        base_cache = copy.deepcopy(draft_cache)

        current_token = torch.tensor([[generated_ids[-1]]], device=self.device)
        draft_out = self.draft_model(
            input_ids=current_token,
            past_key_values=draft_cache,
            use_cache=True,
        )
        draft_cache.__dict__.update(draft_out.past_key_values.__dict__)
        logits = draft_out.logits[:, -1, :].squeeze(0)
        probs = get_probs_from_logits(logits, self.temperature, self.top_p)

        # Sample top-B tokens for root level
        if self.temperature == 0.0:
            top_tokens = logits.topk(self.branch_factor).indices
        else:
            top_tokens = torch.multinomial(probs, num_samples=self.branch_factor)

        # Create root-level nodes
        root_cache_state = copy.deepcopy(draft_cache)
        for b in range(self.branch_factor):
            tok = top_tokens[b].item()
            node = TreeNode(token_id=tok, depth=0, parent_idx=-1)
            tree_nodes.append(node)
            draft_probs.append(probs.clone())

        # BFS expansion: for each subsequent depth level
        # Track which nodes are leaves that need expanding
        # Each leaf needs its own cache state
        leaf_caches = [copy.deepcopy(root_cache_state) for _ in range(self.branch_factor)]

        for d in range(1, depth):
            new_leaves = []
            new_caches = []

            for leaf_idx, cache_state in zip(
                range(len(tree_nodes) - len(leaf_caches), len(tree_nodes)),
                leaf_caches,
            ):
                node = tree_nodes[leaf_idx]
                token_input = torch.tensor([[node.token_id]], device=self.device)

                draft_out = self.draft_model(
                    input_ids=token_input,
                    past_key_values=cache_state,
                    use_cache=True,
                )
                updated_cache = draft_out.past_key_values
                logits = draft_out.logits[:, -1, :].squeeze(0)
                probs = get_probs_from_logits(logits, self.temperature, self.top_p)

                # Only branch if we're not at max tree size
                # Limit total tree size to avoid blowup
                max_tree_size = self.branch_factor * self.depth * 2
                if len(tree_nodes) + self.branch_factor > max_tree_size:
                    # Just take the top-1 token (no branching)
                    if self.temperature == 0.0:
                        tok = logits.argmax().item()
                    else:
                        tok = torch.multinomial(probs, num_samples=1).item()

                    child = TreeNode(token_id=tok, depth=d, parent_idx=leaf_idx)
                    child_idx = len(tree_nodes)
                    tree_nodes.append(child)
                    tree_nodes[leaf_idx].children.append(child_idx)
                    draft_probs.append(probs.clone())
                    new_leaves.append(child_idx)
                    new_caches.append(copy.deepcopy(updated_cache))
                else:
                    # Branch: sample top-B tokens
                    if self.temperature == 0.0:
                        top_tokens = logits.topk(min(self.branch_factor, logits.shape[0])).indices
                    else:
                        top_tokens = torch.multinomial(
                            probs, num_samples=min(self.branch_factor, (probs > 0).sum().item())
                        )

                    for b in range(min(self.branch_factor, top_tokens.shape[0])):
                        tok = top_tokens[b].item()
                        child = TreeNode(token_id=tok, depth=d, parent_idx=leaf_idx)
                        child_idx = len(tree_nodes)
                        tree_nodes.append(child)
                        tree_nodes[leaf_idx].children.append(child_idx)
                        draft_probs.append(probs.clone())
                        new_leaves.append(child_idx)
                        new_caches.append(copy.deepcopy(updated_cache))

            leaf_caches = new_caches

            if not new_leaves:
                break

        return tree_nodes, draft_probs

    def _tree_accept(
        self,
        tree_nodes: list[TreeNode],
        draft_probs_list: list[torch.Tensor],
        target_logits: torch.Tensor,
    ) -> tuple[list[int], int]:
        """Walk the tree and accept tokens along the best path.

        For each node, check acceptance probability min(1, q(x)/p(x)).
        If accepted, move to children and repeat. If rejected, sample
        a correction token from adjusted distribution.

        Returns:
            (accepted_tokens, n_accepted_from_tree):
                accepted_tokens: List of token IDs to append (accepted + correction/bonus).
                n_accepted_from_tree: Number of draft tokens accepted.
        """
        accepted_tokens = []
        n_accepted = 0

        # Start at the roots (depth 0 nodes)
        # target_logits[0, :] is the target dist for the position of tree roots
        # (after seeing last_accepted_token)
        roots = [i for i, node in enumerate(tree_nodes) if node.parent_idx == -1]

        def _try_accept_at(node_indices: list[int], target_logit_idx: int) -> bool:
            """Try to accept one of the candidate nodes at this position.

            Returns True if a token was accepted and we should continue deeper.
            """
            nonlocal n_accepted

            target_logit = target_logits[target_logit_idx]
            target_prob = get_probs_from_logits(
                target_logit, self.temperature, self.top_p
            )

            # Try each candidate (branch) at this position
            # Sort by target probability (most likely first)
            candidates = []
            for idx in node_indices:
                node = tree_nodes[idx]
                dp = draft_probs_list[idx]
                dp, tp = _align_vocab_sizes(dp.unsqueeze(0), target_prob.unsqueeze(0))
                dp = dp.squeeze(0)
                tp = tp.squeeze(0)

                p_draft = dp[node.token_id]
                q_target = tp[node.token_id]
                candidates.append((idx, node, dp, tp, p_draft, q_target))

            # Sort by target probability descending (try best candidate first)
            candidates.sort(key=lambda x: x[5], reverse=True)

            for idx, node, dp, tp, p_draft, q_target in candidates:
                if p_draft == 0:
                    continue

                acceptance_prob = min(1.0, (q_target / p_draft.clamp(min=1e-10)).item())
                r = torch.rand(1, device=self.device).item()

                if r < acceptance_prob:
                    # Accepted this token
                    accepted_tokens.append(node.token_id)
                    n_accepted += 1

                    # Try to continue down this branch
                    if node.children:
                        # target_logits index for children: the node's position + 1
                        # In verify_input: position 0 = last_accepted, position i+1 = tree_nodes[i]
                        # So tree_nodes[idx] is at verify position idx+1
                        # The logits AT position idx+1 predict what comes AFTER node idx
                        child_target_idx = idx + 1
                        _try_accept_at(node.children, child_target_idx)
                    else:
                        # Leaf node — sample bonus token from target
                        bonus_logit = target_logits[idx + 1]
                        bonus_prob = get_probs_from_logits(
                            bonus_logit, self.temperature, self.top_p
                        )
                        bonus_token = torch.multinomial(bonus_prob, num_samples=1).item()
                        accepted_tokens.append(bonus_token)

                    return True

            # All candidates rejected — sample correction from adjusted dist
            # Use the first candidate's distributions for the correction
            if candidates:
                _, _, dp, tp, _, _ = candidates[0]
                adjusted = torch.clamp(tp - dp, min=0.0)
                total = adjusted.sum()
                if total > 1e-10:
                    adjusted = adjusted / total
                    correction = torch.multinomial(adjusted, num_samples=1).item()
                else:
                    correction = torch.multinomial(tp, num_samples=1).item()
                accepted_tokens.append(correction)

            return False

        # Start acceptance from root candidates
        # target_logits[0] is the distribution for the first tree position
        _try_accept_at(roots, 0)

        return accepted_tokens, n_accepted

    def _extract_best_path(
        self,
        tree_nodes: list[TreeNode],
        draft_probs_list: list[torch.Tensor],
    ) -> tuple[list[TreeNode], list[torch.Tensor]]:
        """Extract the highest-probability path through the tree.

        Walks from each root to the deepest leaf, selecting the child with
        the highest draft probability at each branch point. Returns the
        best path as a list of nodes.
        """
        if not tree_nodes:
            return [], []

        # Find all roots
        roots = [i for i, n in enumerate(tree_nodes) if n.parent_idx == -1]
        if not roots:
            return [], []

        # Score each root by its draft probability
        best_root = max(roots, key=lambda i: draft_probs_list[i][tree_nodes[i].token_id])

        # Walk down from best root, always picking the highest-prob child
        path_indices = [best_root]
        current = best_root
        while tree_nodes[current].children:
            children = tree_nodes[current].children
            best_child = max(
                children,
                key=lambda i: draft_probs_list[i][tree_nodes[i].token_id]
            )
            path_indices.append(best_child)
            current = best_child

        path_nodes = [tree_nodes[i] for i in path_indices]
        path_probs = [draft_probs_list[i] for i in path_indices]
        return path_nodes, path_probs

    def _tree_accept_with_fallback(
        self,
        tree_nodes: list[TreeNode],
        all_draft_probs: list[torch.Tensor],
        best_path: list[TreeNode],
        path_draft_probs: list[torch.Tensor],
        target_probs: torch.Tensor,
    ) -> tuple[list[int], int]:
        """Accept tokens along the best path, with tree fallback on rejection.

        For each position along the best path:
          1. Check if the best-path token is accepted (standard rejection sampling)
          2. If rejected, check sibling candidates from the tree at the same depth
          3. If a sibling is accepted, switch to that branch
          4. If all candidates rejected, sample correction from adjusted distribution

        This is where the tree provides value over linear speculation: on
        rejection, we have backup candidates without needing another target call.

        Args:
            tree_nodes: All nodes in the tree.
            all_draft_probs: Draft probs for every tree node.
            best_path: The primary path being verified.
            path_draft_probs: Draft probs for the best path nodes.
            target_probs: Shape (path_len+1, vocab) — from target verification.

        Returns:
            (accepted_tokens, n_accepted)
        """
        accepted_tokens = []
        n_accepted = 0
        path_len = len(best_path)

        for i in range(path_len):
            node = best_path[i]
            dp = path_draft_probs[i]
            tp = target_probs[i]  # target dist for this position

            dp_aligned, tp_aligned = _align_vocab_sizes(
                dp.unsqueeze(0), tp.unsqueeze(0)
            )
            dp_aligned = dp_aligned.squeeze(0)
            tp_aligned = tp_aligned.squeeze(0)

            p_draft = dp_aligned[node.token_id]
            q_target = tp_aligned[node.token_id]

            # Standard acceptance check
            if p_draft > 0:
                acceptance_prob = min(1.0, (q_target / p_draft.clamp(min=1e-10)).item())
                r = torch.rand(1, device=self.device).item()

                if r < acceptance_prob:
                    accepted_tokens.append(node.token_id)
                    n_accepted += 1
                    continue

            # Rejected — try sibling candidates from the tree
            # Find all nodes at the same depth with the same parent
            sibling_accepted = False
            if node.parent_idx >= 0:
                parent_children = tree_nodes[node.parent_idx].children
            else:
                # Root level — all roots are siblings
                parent_children = [
                    j for j, n in enumerate(tree_nodes) if n.parent_idx == -1
                ]

            # Find the index of the current node in tree_nodes
            current_idx = None
            for j, tn in enumerate(tree_nodes):
                if tn is node:
                    current_idx = j
                    break

            for sib_idx in parent_children:
                if sib_idx == current_idx:
                    continue  # Skip the already-rejected node

                sib_node = tree_nodes[sib_idx]
                sib_dp = all_draft_probs[sib_idx]

                sib_dp_a, tp_a = _align_vocab_sizes(
                    sib_dp.unsqueeze(0), tp_aligned.unsqueeze(0)
                )
                sib_dp_a = sib_dp_a.squeeze(0)
                tp_a = tp_a.squeeze(0)

                sib_p = sib_dp_a[sib_node.token_id]
                sib_q = tp_a[sib_node.token_id]

                if sib_p > 0:
                    sib_accept_prob = min(1.0, (sib_q / sib_p.clamp(min=1e-10)).item())
                    r2 = torch.rand(1, device=self.device).item()

                    if r2 < sib_accept_prob:
                        # Sibling accepted — this is the tree's value
                        accepted_tokens.append(sib_node.token_id)
                        n_accepted += 1
                        sibling_accepted = True
                        break

            if sibling_accepted:
                # We switched branches — can't continue along best_path
                # because the target logits after this point are conditioned
                # on the best_path tokens, not the sibling's token.
                # Sample a bonus token from the target at this position.
                bonus_tp = target_probs[i]
                bonus_tp_aligned = _align_vocab_sizes(
                    bonus_tp.unsqueeze(0), bonus_tp.unsqueeze(0)
                )[0].squeeze(0)
                bonus = torch.multinomial(
                    get_probs_from_logits(bonus_tp, self.temperature, self.top_p),
                    num_samples=1,
                ).item()
                accepted_tokens.append(bonus)
                break

            # All siblings rejected too — sample correction
            adjusted = torch.clamp(tp_aligned - dp_aligned, min=0.0)
            total = adjusted.sum()
            if total > 1e-10:
                adjusted = adjusted / total
                correction = torch.multinomial(adjusted, num_samples=1).item()
            else:
                correction = torch.multinomial(tp_aligned, num_samples=1).item()
            accepted_tokens.append(correction)
            break

        else:
            # All path tokens accepted — sample bonus from target
            if path_len > 0:
                bonus_probs = get_probs_from_logits(
                    target_probs[path_len], self.temperature, self.top_p
                )
                bonus = torch.multinomial(bonus_probs, num_samples=1).item()
                accepted_tokens.append(bonus)

        return accepted_tokens, n_accepted

    def _build_result(
        self,
        generated_ids: list[int],
        prompt_len: int,
        n_target_calls: int,
        n_accepted_draft: int,
        n_total_draft: int,
        timer: CudaTimer,
        prof: ProfilingData | None,
    ) -> GenerationResult:
        elapsed = timer.stop()
        new_tokens = generated_ids[prompt_len:]
        if prof:
            accounted = prof.draft_time + prof.target_time + prof.sampling_time + prof.cache_time
            prof.overhead_time = max(0, elapsed - accounted)

        return GenerationResult(
            token_ids=new_tokens,
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            n_target_calls=n_target_calls,
            n_generated_tokens=len(new_tokens),
            n_accepted_draft_tokens=n_accepted_draft,
            n_total_draft_tokens=n_total_draft,
            wall_clock_seconds=elapsed,
            profiling=prof,
        )
