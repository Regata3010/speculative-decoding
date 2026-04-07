"""Core speculative decoding loop (Phase 2 — optimized).

Optimizations over Phase 1:
  - Per-phase profiling (draft, target, sampling, cache, overhead)
  - Vectorized probability computation (batch get_probs_from_logits)
  - Pre-allocated tensor for verify_input (avoids per-step allocation)
  - Reduced Python overhead in hot loop

Cache invariant: at the start of each speculative iteration, both caches
contain KV entries for all tokens in generated_ids EXCEPT the last one.
The last token is fed as part of the next forward pass.
"""

import time

import torch

from src.kv_cache_manager import create_cache, get_cache_length, rollback_cache
from src.sampling import get_probs_from_logits, rejection_sample, sample_from_logits
from src.utils import CudaTimer, GenerationResult, ProfilingData


class SpeculativeDecoder:
    """Speculative decoding engine with per-phase profiling."""

    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        K: int = 5,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.K = K
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
        """Generate tokens using speculative decoding.

        Args:
            input_ids: Shape (1, seq_len) — tokenized prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            profile: If True, record per-phase timing breakdown.

        Returns:
            GenerationResult with generated tokens, metrics, and profiling data.
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

        # Per-position acceptance tracking
        position_accepted = [0] * self.K
        position_proposed = [0] * self.K

        # Profiling accumulators
        prof = ProfilingData() if profile else None
        _sync = torch.cuda.synchronize if self.device.type == "cuda" else lambda: None

        # Pre-allocate verify input buffer (K+1 tokens max)
        verify_buffer = torch.zeros(1, self.K + 1, dtype=torch.long, device=self.device)

        # === Prefill: process prompt through both models ===
        prompt_tensor = input_ids

        if prof:
            _sync()
            t0 = time.perf_counter()

        target_out = self.target_model(
            input_ids=prompt_tensor, past_key_values=target_cache, use_cache=True
        )
        target_cache = target_out.past_key_values
        n_target_calls += 1

        if prof:
            _sync()
            prof.target_time += time.perf_counter() - t0
            t0 = time.perf_counter()

        draft_out = self.draft_model(
            input_ids=prompt_tensor, past_key_values=draft_cache, use_cache=True
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

        # === Main speculative decoding loop ===
        while len(generated_ids) - prompt_len < max_new_tokens:
            tokens_remaining = max_new_tokens - (len(generated_ids) - prompt_len)
            k = min(self.K, tokens_remaining)
            if k <= 0:
                break

            if prof:
                _sync()
                t_loop_start = time.perf_counter()

            # --- Step 1: Draft generates K tokens autoregressively ---
            if prof:
                _sync()
                t0 = time.perf_counter()

            draft_token_ids, draft_logits_list = self._draft_generate(
                generated_ids, draft_cache, k
            )

            if prof:
                _sync()
                prof.draft_time += time.perf_counter() - t0

            # Vectorized: convert all draft logits to probs in one batch call
            draft_logits_stacked = torch.stack(draft_logits_list)  # (k, vocab)
            draft_probs = get_probs_from_logits(
                draft_logits_stacked, self.temperature, self.top_p
            )

            # --- Step 2: Target verifies all K draft tokens in one pass ---
            # Write into pre-allocated buffer instead of creating new tensor
            verify_buffer[0, 0] = generated_ids[-1]
            for j in range(k):
                verify_buffer[0, j + 1] = draft_token_ids[j]
            verify_input = verify_buffer[:, :k + 1]

            if prof:
                _sync()
                t0 = time.perf_counter()

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

            # Vectorized: convert all target logits to probs in one batch call
            all_target_logits = target_out.logits.squeeze(0)  # (k+1, vocab)
            target_probs = get_probs_from_logits(
                all_target_logits, self.temperature, self.top_p
            )

            # --- Step 3: Rejection sampling (vectorized) ---
            if prof:
                _sync()
                t0 = time.perf_counter()

            draft_tokens_tensor = torch.tensor(draft_token_ids, device=self.device)
            n_accepted, next_token = rejection_sample(
                draft_probs, target_probs, draft_tokens_tensor
            )

            if prof:
                _sync()
                prof.sampling_time += time.perf_counter() - t0

            n_total_draft += k
            n_accepted_draft += n_accepted

            # Track per-position acceptance
            for pos in range(k):
                if pos < len(position_proposed):
                    position_proposed[pos] += 1
                    if pos < n_accepted:
                        position_accepted[pos] += 1
                else:
                    # k > self.K shouldn't happen, but be safe
                    break

            # Append accepted draft tokens
            hit_eos = False
            for i in range(n_accepted):
                generated_ids.append(draft_token_ids[i])
                if draft_token_ids[i] == eos_token_id:
                    hit_eos = True
                    break

            if not hit_eos:
                generated_ids.append(next_token.item())
                if next_token.item() == eos_token_id:
                    hit_eos = True

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

        # Compute overhead as residual
        if prof:
            total_elapsed = timer.stop()
            accounted = prof.draft_time + prof.target_time + prof.sampling_time + prof.cache_time
            prof.overhead_time = max(0, total_elapsed - accounted)
            return GenerationResult(
                token_ids=generated_ids[prompt_len:],
                text=self.tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True),
                n_target_calls=n_target_calls,
                n_generated_tokens=len(generated_ids) - prompt_len,
                n_accepted_draft_tokens=n_accepted_draft,
                n_total_draft_tokens=n_total_draft,
                wall_clock_seconds=total_elapsed,
                profiling=prof,
                position_accepted=position_accepted[:self.K],
                position_proposed=position_proposed[:self.K],
            )

        return self._build_result(
            generated_ids, prompt_len, n_target_calls,
            n_accepted_draft, n_total_draft, timer, prof,
            position_accepted, position_proposed,
        )

    def _draft_generate(
        self,
        generated_ids: list[int],
        draft_cache,
        k: int,
    ) -> tuple[list[int], list[torch.Tensor]]:
        """Generate k draft tokens, syncing cache if needed."""
        expected_cache_len = len(generated_ids) - 1
        current_cache_len = get_cache_length(draft_cache)

        if current_cache_len < expected_cache_len:
            missing_tokens = generated_ids[current_cache_len:expected_cache_len]
            if missing_tokens:
                missing_input = torch.tensor([missing_tokens], device=self.device)
                draft_out = self.draft_model(
                    input_ids=missing_input,
                    past_key_values=draft_cache,
                    use_cache=True,
                )
                draft_cache.__dict__.update(draft_out.past_key_values.__dict__)

        draft_token_ids = []
        draft_logits_list = []
        current_token = torch.tensor(
            [[generated_ids[-1]]], device=self.device
        )

        for _ in range(k):
            draft_out = self.draft_model(
                input_ids=current_token,
                past_key_values=draft_cache,
                use_cache=True,
            )
            draft_cache.__dict__.update(draft_out.past_key_values.__dict__)

            logits = draft_out.logits[:, -1, :].squeeze(0)
            draft_logits_list.append(logits)

            token = sample_from_logits(logits, self.temperature, self.top_p)
            draft_token_ids.append(token.item())
            current_token = token.unsqueeze(0).unsqueeze(0)

        return draft_token_ids, draft_logits_list

    def _build_result(
        self,
        generated_ids: list[int],
        prompt_len: int,
        n_target_calls: int,
        n_accepted_draft: int,
        n_total_draft: int,
        timer: CudaTimer,
        prof: ProfilingData | None,
        position_accepted: list[int] | None = None,
        position_proposed: list[int] | None = None,
    ) -> GenerationResult:
        elapsed = timer.stop()
        new_tokens = generated_ids[prompt_len:]

        # Compute overhead as residual
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
            position_accepted=position_accepted or [],
            position_proposed=position_proposed or [],
        )
