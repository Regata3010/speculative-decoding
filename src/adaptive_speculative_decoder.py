"""Adaptive speculative decoding with dynamic K adjustment.

Extends the base speculative decoder with a controller that adjusts K
(the number of draft tokens per step) based on recent acceptance rates.

When the draft model is performing well (high acceptance), K increases
to extract more tokens per target call. When acceptance drops, K decreases
to avoid wasting compute on rejected tokens.

This is what production systems do — static K leaves performance on the table.
"""

import torch
from collections import deque

from src.kv_cache_manager import create_cache, get_cache_length, rollback_cache
from src.sampling import get_probs_from_logits, rejection_sample, sample_from_logits
from src.utils import CudaTimer, GenerationResult


class AdaptiveKController:
    """Adjusts K based on a sliding window of recent acceptance rates.

    Strategy:
      - Track acceptance rate over the last `window_size` speculative steps
      - If rate > high_threshold: increase K (draft model is reliable)
      - If rate < low_threshold: decrease K (draft model is struggling)
      - Otherwise: hold K steady

    This adapts to changing text characteristics mid-generation.
    For example, code completion may have high acceptance on boilerplate
    but low acceptance on novel logic.
    """

    def __init__(
        self,
        initial_k: int = 5,
        min_k: int = 1,
        max_k: int = 20,
        window_size: int = 8,
        high_threshold: float = 0.8,
        low_threshold: float = 0.4,
        step_size: int = 1,
    ):
        self.k = initial_k
        self.min_k = min_k
        self.max_k = max_k
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.step_size = step_size
        self._history: deque[float] = deque(maxlen=window_size)

    def update(self, n_accepted: int, n_proposed: int):
        """Record the result of a speculative step and adjust K."""
        if n_proposed > 0:
            rate = n_accepted / n_proposed
            self._history.append(rate)

        if len(self._history) < 2:
            return  # Not enough data yet

        avg_rate = sum(self._history) / len(self._history)

        if avg_rate > self.high_threshold:
            self.k = min(self.k + self.step_size, self.max_k)
        elif avg_rate < self.low_threshold:
            self.k = max(self.k - self.step_size, self.min_k)

    @property
    def current_k(self) -> int:
        return self.k

    @property
    def recent_acceptance_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)


class AdaptiveSpeculativeDecoder:
    """Speculative decoder with dynamic K adjustment."""

    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        initial_K: int = 5,
        min_K: int = 1,
        max_K: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
        high_threshold: float = 0.8,
        low_threshold: float = 0.4,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.device = next(target_model.parameters()).device

        self.controller = AdaptiveKController(
            initial_k=initial_K,
            min_k=min_K,
            max_k=max_K,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> GenerationResult:
        """Generate tokens using adaptive speculative decoding.

        The number of draft tokens K adjusts dynamically based on
        recent acceptance rates during generation.

        Args:
            input_ids: Shape (1, seq_len) — tokenized prompt.
            max_new_tokens: Maximum number of new tokens to generate.

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
        k_history = []  # Track K values used

        # Reset controller for each generation
        self.controller = AdaptiveKController(
            initial_k=self.controller.k,
            min_k=self.controller.min_k,
            max_k=self.controller.max_k,
            high_threshold=self.controller.high_threshold,
            low_threshold=self.controller.low_threshold,
        )

        # Prefill
        prompt_tensor = input_ids
        target_out = self.target_model(
            input_ids=prompt_tensor, past_key_values=target_cache, use_cache=True
        )
        target_cache = target_out.past_key_values
        n_target_calls += 1

        draft_out = self.draft_model(
            input_ids=prompt_tensor, past_key_values=draft_cache, use_cache=True
        )
        draft_cache = draft_out.past_key_values

        # Sample first token
        first_token = sample_from_logits(
            target_out.logits[:, -1, :].squeeze(0), self.temperature, self.top_p
        )
        generated_ids.append(first_token.item())

        if first_token.item() == eos_token_id:
            return self._build_result(
                generated_ids, prompt_len, n_target_calls,
                n_accepted_draft, n_total_draft, k_history, timer,
            )

        # Main loop
        while len(generated_ids) - prompt_len < max_new_tokens:
            tokens_remaining = max_new_tokens - (len(generated_ids) - prompt_len)
            k = min(self.controller.current_k, tokens_remaining)
            if k <= 0:
                break

            k_history.append(k)

            # Draft generate
            draft_token_ids, draft_logits_list = self._draft_generate(
                generated_ids, draft_cache, k
            )

            draft_logits_stacked = torch.stack(draft_logits_list)
            draft_probs = get_probs_from_logits(
                draft_logits_stacked, self.temperature, self.top_p
            )

            # Target verify
            verify_tokens = [generated_ids[-1]] + draft_token_ids
            verify_input = torch.tensor([verify_tokens], device=self.device)

            expected_target_len = len(generated_ids) - 1
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

            target_out = self.target_model(
                input_ids=verify_input,
                past_key_values=target_cache,
                use_cache=True,
            )
            target_cache = target_out.past_key_values
            n_target_calls += 1

            all_target_logits = target_out.logits.squeeze(0)
            target_probs = get_probs_from_logits(
                all_target_logits, self.temperature, self.top_p
            )

            # Rejection sampling
            draft_tokens_tensor = torch.tensor(draft_token_ids, device=self.device)
            n_accepted, next_token = rejection_sample(
                draft_probs, target_probs, draft_tokens_tensor
            )

            n_total_draft += k
            n_accepted_draft += n_accepted

            # Update adaptive controller
            self.controller.update(n_accepted, k)

            # Append accepted tokens
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
                    n_accepted_draft, n_total_draft, k_history, timer,
                )

            # Rollback caches
            new_cache_len = len(generated_ids) - 1
            rollback_cache(target_cache, new_cache_len)
            rollback_cache(draft_cache, new_cache_len)

        return self._build_result(
            generated_ids, prompt_len, n_target_calls,
            n_accepted_draft, n_total_draft, k_history, timer,
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
        k_history: list[int],
        timer: CudaTimer,
    ) -> GenerationResult:
        elapsed = timer.stop()
        new_tokens = generated_ids[prompt_len:]
        result = GenerationResult(
            token_ids=new_tokens,
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            n_target_calls=n_target_calls,
            n_generated_tokens=len(new_tokens),
            n_accepted_draft_tokens=n_accepted_draft,
            n_total_draft_tokens=n_total_draft,
            wall_clock_seconds=elapsed,
        )
        # Attach K history as extra metadata (accessible but doesn't break interface)
        result.k_history = k_history
        return result
