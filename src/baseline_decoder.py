"""Standard autoregressive decoding baseline.

Implements the same GenerationResult interface as SpeculativeDecoder
for apples-to-apples comparison. Built from scratch (not using HF .generate())
to ensure identical overhead in timing measurements.
"""

import torch

from src.kv_cache_manager import create_cache
from src.sampling import sample_from_logits
from src.utils import CudaTimer, GenerationResult


class BaselineDecoder:
    """Standard autoregressive decoder using only the target model."""

    def __init__(
        self,
        target_model,
        tokenizer,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.device = next(target_model.parameters()).device

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> GenerationResult:
        """Generate tokens using standard autoregressive decoding.

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

        cache = create_cache()
        n_target_calls = 0

        # Prefill: process prompt
        out = self.target_model(
            input_ids=input_ids, past_key_values=cache, use_cache=True
        )
        cache = out.past_key_values
        logits = out.logits[:, -1, :].squeeze(0)
        n_target_calls += 1

        # Autoregressive loop: one token per forward pass
        for _ in range(max_new_tokens):
            token = sample_from_logits(logits, self.temperature, self.top_p)
            generated_ids.append(token.item())

            if token.item() == eos_token_id:
                break

            # Feed the new token
            token_input = token.unsqueeze(0).unsqueeze(0)
            out = self.target_model(
                input_ids=token_input, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)
            n_target_calls += 1

        elapsed = timer.stop()
        new_tokens = generated_ids[prompt_len:]
        return GenerationResult(
            token_ids=new_tokens,
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            n_target_calls=n_target_calls,
            n_generated_tokens=len(new_tokens),
            n_accepted_draft_tokens=0,
            n_total_draft_tokens=0,
            wall_clock_seconds=elapsed,
        )
