"""HuggingFace built-in assisted generation baseline.

Uses transformers' native assisted generation (model.generate with
assistant_model) as a third comparison arm alongside our from-scratch
speculative decoder and the autoregressive baseline.

This shows how our implementation compares to an optimized library
implementation with the same models.

Note: HF's assisted generation can fail with certain model combinations
due to vocab size mismatches or internal bugs in the cross-tokenizer
translation layer. When this happens, the decoder falls back gracefully
and returns an empty result.
"""

import warnings

import torch

from src.utils import CudaTimer, GenerationResult


class HFAssistedDecoder:
    """Wrapper around HuggingFace's built-in assisted generation."""

    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.device = next(target_model.parameters()).device
        self._warned = False

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> GenerationResult:
        """Generate tokens using HF's built-in assisted generation.

        Falls back to an empty result if HF's internal assisted generation
        fails (e.g., due to vocab size mismatches between models).
        """
        timer = CudaTimer(self.device)
        timer.start()

        input_ids = input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        try:
            # Try without explicit tokenizers first (works for same-vocab models)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "assistant_model": self.draft_model,
                "do_sample": self.temperature > 0,
            }

            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
                if self.top_p < 1.0:
                    gen_kwargs["top_p"] = self.top_p

            output = self.target_model.generate(input_ids, **gen_kwargs)

            elapsed = timer.stop()
            new_token_ids = output[0, prompt_len:].tolist()
            text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

            return GenerationResult(
                token_ids=new_token_ids,
                text=text,
                n_target_calls=0,
                n_generated_tokens=len(new_token_ids),
                n_accepted_draft_tokens=0,
                n_total_draft_tokens=0,
                wall_clock_seconds=elapsed,
            )

        except (ValueError, AttributeError, RuntimeError) as e:
            elapsed = timer.stop()
            if not self._warned:
                warnings.warn(
                    f"HF assisted generation failed: {e}\n"
                    f"This is a known issue with some model combinations in "
                    f"transformers. HF results will be empty — our speculative "
                    f"decoder and baseline comparisons are unaffected.",
                    stacklevel=2,
                )
                self._warned = True

            return GenerationResult(
                token_ids=[],
                text="",
                n_target_calls=0,
                n_generated_tokens=0,
                n_accepted_draft_tokens=0,
                n_total_draft_tokens=0,
                wall_clock_seconds=elapsed,
            )
