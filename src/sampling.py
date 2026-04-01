"""Rejection sampling for speculative decoding.

Implements the acceptance/rejection scheme from:
  Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)

The key guarantee: the output distribution is identical to sampling from the
target model alone, regardless of draft model quality.
"""

import torch
import torch.nn.functional as F


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample a token from logits with temperature and nucleus (top-p) sampling.

    Args:
        logits: Shape (vocab_size,) — raw logits for a single position.
        temperature: Sampling temperature. 0.0 = greedy.
        top_p: Nucleus sampling threshold. 1.0 = no filtering.

    Returns:
        Scalar tensor with the sampled token ID.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_p < 1.0:
        logits = _apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def get_probs_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Convert logits to a probability distribution with temperature and top-p.

    Args:
        logits: Shape (..., vocab_size).
        temperature: Sampling temperature. Clamped to 1e-6 minimum.
        top_p: Nucleus sampling threshold.

    Returns:
        Probability tensor of same shape.
    """
    if temperature == 0.0:
        # For greedy: one-hot on argmax
        indices = logits.argmax(dim=-1, keepdim=True)
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, indices, 1.0)
        return probs

    logits = logits / max(temperature, 1e-6)

    if top_p < 1.0:
        logits = _apply_top_p(logits, top_p)

    return F.softmax(logits, dim=-1)


def _align_vocab_sizes(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the smaller tensor with zeros to match vocab sizes.

    Some model families (e.g., Qwen 2.5) have slightly different vocab sizes
    between the target and draft models despite sharing the same tokenizer.
    The extra tokens in the larger vocab are never sampled by the smaller
    model, so padding with zero probability is correct.
    """
    draft_vocab = draft_probs.shape[-1]
    target_vocab = target_probs.shape[-1]

    if draft_vocab == target_vocab:
        return draft_probs, target_probs

    max_vocab = max(draft_vocab, target_vocab)

    if draft_vocab < max_vocab:
        pad = torch.zeros(
            *draft_probs.shape[:-1], max_vocab - draft_vocab,
            device=draft_probs.device, dtype=draft_probs.dtype,
        )
        draft_probs = torch.cat([draft_probs, pad], dim=-1)

    if target_vocab < max_vocab:
        pad = torch.zeros(
            *target_probs.shape[:-1], max_vocab - target_vocab,
            device=target_probs.device, dtype=target_probs.dtype,
        )
        target_probs = torch.cat([target_probs, pad], dim=-1)

    return draft_probs, target_probs


def rejection_sample(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Apply rejection sampling to a sequence of K draft tokens (vectorized).

    Computes all acceptance probabilities in one GPU operation, generates all
    random values in one call, and finds the first rejection point without
    a Python loop. Only falls back to Python for the final resampling step.

    Args:
        draft_probs: Shape (K, vocab_size) — draft model probabilities.
        target_probs: Shape (K+1, vocab_size) — target model probabilities
                      (K verification positions + 1 bonus position).
        draft_token_ids: Shape (K,) — the draft tokens to verify.

    Returns:
        (n_accepted, next_token):
            n_accepted: Number of draft tokens accepted (0 to K).
            next_token: The next token to append — either a resampled
                       correction token or a bonus token.
    """
    draft_probs, target_probs = _align_vocab_sizes(draft_probs, target_probs)

    K = draft_token_ids.shape[0]
    device = draft_token_ids.device

    # Vectorized: gather draft and target probs for all K draft tokens at once
    # draft_token_ids shape: (K,) → index into (K, vocab) → (K,)
    p = draft_probs.gather(1, draft_token_ids.unsqueeze(1)).squeeze(1)  # (K,)
    q = target_probs[:K].gather(1, draft_token_ids.unsqueeze(1)).squeeze(1)  # (K,)

    # Acceptance probabilities: min(1, q/p), handle p=0 by rejecting
    acceptance_probs = torch.where(
        p > 0,
        torch.clamp(q / p.clamp(min=1e-10), max=1.0),
        torch.zeros_like(p),  # p=0 → always reject
    )  # (K,)

    # Generate all random values in one call
    rand_vals = torch.rand(K, device=device)

    # Find first rejection: where rand >= acceptance_prob
    rejected = rand_vals >= acceptance_probs  # (K,) bool

    if not rejected.any():
        # All K tokens accepted — sample bonus token from target at position K
        bonus_probs = target_probs[K]
        next_token = torch.multinomial(bonus_probs, num_samples=1).squeeze(-1)
        return K, next_token

    # Find first rejection index
    first_reject = rejected.long().argmax().item()

    # Sample correction token from adjusted distribution at rejection point
    next_token = _sample_adjusted(
        draft_probs[first_reject], target_probs[first_reject]
    )
    return first_reject, next_token


def _sample_adjusted(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """Sample from the adjusted distribution: norm(max(0, q - p)).

    This ensures the output distribution matches the target model exactly.

    Args:
        draft_probs: Shape (vocab_size,) — draft distribution at this position.
        target_probs: Shape (vocab_size,) — target distribution at this position.

    Returns:
        Scalar tensor with the sampled token ID.
    """
    adjusted = torch.clamp(target_probs - draft_probs, min=0.0)
    total = adjusted.sum()

    if total <= 1e-10:
        # Fallback: distributions are nearly identical, sample from target
        return torch.multinomial(target_probs, num_samples=1).squeeze(-1)

    adjusted = adjusted / total
    return torch.multinomial(adjusted, num_samples=1).squeeze(-1)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits.

    Args:
        logits: Shape (..., vocab_size). Supports batched input.
        top_p: Cumulative probability threshold.

    Returns:
        Filtered logits with low-probability tokens set to -inf.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    # Shift right so the first token above threshold is kept
    sorted_mask = (cumulative_probs - sorted_probs) >= top_p
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # Scatter back to original indices
    result = torch.zeros_like(logits)
    result.scatter_(-1, sorted_indices, sorted_logits)
    return result
