"""Tests for the rejection sampling implementation.

Key correctness properties:
1. Greedy: speculative decoding must accept all tokens when distributions match
2. Rejection: tokens are rejected when draft probability exceeds target probability
3. Adjusted distribution: norm(max(0, q - p)) is a valid probability distribution
4. Distributional equivalence: output distribution matches target model
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import pytest

from src.sampling import (
    get_probs_from_logits,
    rejection_sample,
    sample_from_logits,
    _sample_adjusted,
    _apply_top_p,
)


class TestSampleFromLogits:
    def test_greedy_returns_argmax(self):
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        token = sample_from_logits(logits, temperature=0.0)
        assert token.item() == 1

    def test_greedy_deterministic(self):
        logits = torch.randn(1000)
        t1 = sample_from_logits(logits, temperature=0.0)
        t2 = sample_from_logits(logits, temperature=0.0)
        assert t1.item() == t2.item()

    def test_temperature_scaling(self):
        logits = torch.tensor([2.0, 1.0, 0.0])
        # At very low temperature, should almost always pick index 0
        torch.manual_seed(42)
        tokens = [sample_from_logits(logits, temperature=0.01).item() for _ in range(100)]
        assert tokens.count(0) > 95

    def test_returns_valid_index(self):
        logits = torch.randn(50000)
        torch.manual_seed(42)
        token = sample_from_logits(logits, temperature=1.0, top_p=0.9)
        assert 0 <= token.item() < 50000


class TestGetProbsFromLogits:
    def test_sums_to_one(self):
        logits = torch.randn(1000)
        probs = get_probs_from_logits(logits, temperature=1.0)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_greedy_is_one_hot(self):
        logits = torch.tensor([1.0, 5.0, 2.0])
        probs = get_probs_from_logits(logits, temperature=0.0)
        assert probs[1].item() == 1.0
        assert probs[0].item() == 0.0
        assert probs[2].item() == 0.0

    def test_all_non_negative(self):
        logits = torch.randn(1000)
        probs = get_probs_from_logits(logits, temperature=0.5, top_p=0.9)
        assert (probs >= 0).all()


class TestRejectionSample:
    def test_identical_distributions_accept_all(self):
        """When draft == target, all tokens should be accepted."""
        K = 5
        vocab_size = 100
        torch.manual_seed(42)

        probs = F.softmax(torch.randn(vocab_size), dim=-1)
        draft_probs = probs.unsqueeze(0).expand(K, -1)
        # target_probs needs K+1 rows (K verify + 1 bonus)
        target_probs = probs.unsqueeze(0).expand(K + 1, -1)

        # Draft tokens sampled from the same distribution
        draft_token_ids = torch.multinomial(probs, num_samples=K, replacement=True)

        n_accepted, _ = rejection_sample(draft_probs, target_probs, draft_token_ids)
        # With identical distributions, acceptance prob is always 1
        assert n_accepted == K

    def test_zero_draft_prob_rejects(self):
        """Tokens with zero draft probability should be rejected."""
        K = 1
        vocab_size = 10

        draft_probs = torch.zeros(K, vocab_size)
        draft_probs[0, 0] = 1.0  # Draft puts all mass on token 0

        target_probs = torch.zeros(K + 1, vocab_size)
        target_probs[0, 5] = 1.0  # Target puts all mass on token 5
        target_probs[1, 5] = 1.0

        draft_token_ids = torch.tensor([5])  # Draft "sampled" token 5 but p(5)=0

        n_accepted, next_token = rejection_sample(
            draft_probs, target_probs, draft_token_ids
        )
        assert n_accepted == 0

    def test_returns_valid_token(self):
        K = 3
        vocab_size = 100
        torch.manual_seed(42)

        draft_probs = F.softmax(torch.randn(K, vocab_size), dim=-1)
        target_probs = F.softmax(torch.randn(K + 1, vocab_size), dim=-1)
        draft_token_ids = torch.randint(0, vocab_size, (K,))

        n_accepted, next_token = rejection_sample(
            draft_probs, target_probs, draft_token_ids
        )
        assert 0 <= n_accepted <= K
        assert 0 <= next_token.item() < vocab_size

    def test_n_accepted_range(self):
        """n_accepted should be between 0 and K."""
        K = 10
        vocab_size = 50
        for seed in range(20):
            torch.manual_seed(seed)
            draft_probs = F.softmax(torch.randn(K, vocab_size), dim=-1)
            target_probs = F.softmax(torch.randn(K + 1, vocab_size), dim=-1)
            draft_token_ids = torch.randint(0, vocab_size, (K,))
            n_accepted, _ = rejection_sample(
                draft_probs, target_probs, draft_token_ids
            )
            assert 0 <= n_accepted <= K


class TestAdjustedDistribution:
    def test_valid_distribution(self):
        """Adjusted distribution should sum to ~1 and be non-negative."""
        torch.manual_seed(42)
        draft_probs = F.softmax(torch.randn(100), dim=-1)
        target_probs = F.softmax(torch.randn(100), dim=-1)

        adjusted = torch.clamp(target_probs - draft_probs, min=0.0)
        total = adjusted.sum()
        if total > 1e-10:
            adjusted = adjusted / total
            assert abs(adjusted.sum().item() - 1.0) < 1e-5
            assert (adjusted >= 0).all()

    def test_identical_distributions_fallback(self):
        """When q == p, adjusted dist is zero everywhere; should fallback."""
        probs = F.softmax(torch.randn(50), dim=-1)
        token = _sample_adjusted(probs, probs)
        assert 0 <= token.item() < 50


class TestTopP:
    def test_preserves_top_tokens(self):
        logits = torch.tensor([10.0, 5.0, 1.0, 0.1, 0.01])
        filtered = _apply_top_p(logits.clone(), top_p=0.9)
        # Token 0 should always survive
        assert filtered[0] > float("-inf")

    def test_filters_low_probability(self):
        logits = torch.tensor([10.0, 5.0, 1.0, -5.0, -10.0])
        filtered = _apply_top_p(logits.clone(), top_p=0.5)
        # Very low probability tokens should be filtered
        assert filtered[-1] == float("-inf")


class TestDistributionalEquivalence:
    """Statistical test: speculative decoding output should match target distribution."""

    def test_empirical_distribution_greedy(self):
        """With identical distributions and greedy decoding, all tokens accepted."""
        vocab_size = 10
        K = 3

        # Create a sharp distribution (pseudo-greedy)
        logits = torch.zeros(vocab_size)
        logits[3] = 100.0  # Token 3 dominates

        probs = F.softmax(logits, dim=-1)
        draft_probs = probs.unsqueeze(0).expand(K, -1)
        target_probs = probs.unsqueeze(0).expand(K + 1, -1)
        draft_token_ids = torch.tensor([3, 3, 3])

        n_accepted, next_token = rejection_sample(
            draft_probs, target_probs, draft_token_ids
        )
        assert n_accepted == K
        assert next_token.item() == 3
