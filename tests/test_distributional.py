"""Statistical tests for distributional equivalence.

The core guarantee of speculative decoding: the output distribution is
identical to sampling from the target model alone, regardless of draft
model quality. These tests verify this empirically.

Also includes stress tests for long sequences and memory leak detection.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
from collections import Counter

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from src.speculative_decoder import SpeculativeDecoder
from src.adaptive_speculative_decoder import AdaptiveSpeculativeDecoder
from src.baseline_decoder import BaselineDecoder
from src.utils import set_seed


def _create_tiny_model(vocab_size=64, hidden_size=32, num_layers=2, num_heads=2):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _create_tokenizer(vocab_size=64):
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    tokenizer_model = models.BPE()
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<eos>", "<unk>"],
    )
    tokenizer.train_from_iterator(
        ["The quick brown fox jumps over the lazy dog. " * 50,
         "Pack my box with five dozen liquor jugs. " * 50],
        trainer=trainer,
    )
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    return wrapped


@pytest.fixture(scope="module")
def models_and_tokenizer():
    """Shared fixture — creating models is expensive even when tiny."""
    set_seed(42)
    target = _create_tiny_model()
    draft = _create_tiny_model()
    tokenizer = _create_tokenizer()
    return target, draft, tokenizer


class TestDistributionalEquivalence:
    """Verify that speculative decoding produces the same token distribution
    as standard autoregressive decoding over many samples."""

    def _collect_first_tokens(self, decoder, input_ids, n_samples, max_new=1):
        """Generate n_samples times and collect the first generated token each time."""
        tokens = []
        for i in range(n_samples):
            set_seed(i * 7 + 13)  # Varied seeds
            result = decoder.generate(input_ids.clone(), max_new_tokens=max_new)
            if result.token_ids:
                tokens.append(result.token_ids[0])
        return tokens

    def test_first_token_distribution_matches(self, models_and_tokenizer):
        """Compare first-token distributions across many samples.

        Uses a chi-squared-style comparison: for each token, compute
        the frequency in both distributions and check they're close.
        """
        target, draft, tokenizer = models_and_tokenizer
        input_ids = torch.randint(3, 60, (1, 4))
        n_samples = 500

        spec_decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=3, temperature=1.0
        )
        base_decoder = BaselineDecoder(target, tokenizer, temperature=1.0)

        spec_tokens = self._collect_first_tokens(spec_decoder, input_ids, n_samples)
        base_tokens = self._collect_first_tokens(base_decoder, input_ids, n_samples)

        spec_counter = Counter(spec_tokens)
        base_counter = Counter(base_tokens)

        # Compute total variation distance between the two empirical distributions
        all_tokens = set(spec_counter.keys()) | set(base_counter.keys())
        tvd = 0.0
        for tok in all_tokens:
            p_spec = spec_counter.get(tok, 0) / len(spec_tokens)
            p_base = base_counter.get(tok, 0) / len(base_tokens)
            tvd += abs(p_spec - p_base)
        tvd /= 2.0

        # TVD should be small (< 0.15 for 500 samples is reasonable with noise)
        assert tvd < 0.15, (
            f"Total variation distance too high: {tvd:.3f}\n"
            f"Speculative top-5: {spec_counter.most_common(5)}\n"
            f"Baseline top-5:    {base_counter.most_common(5)}"
        )

    def test_token_frequency_chi_squared(self, models_and_tokenizer):
        """Chi-squared test on token frequencies.

        Null hypothesis: speculative and baseline produce the same distribution.
        We should NOT reject (p-value should be high).
        """
        target, draft, tokenizer = models_and_tokenizer
        input_ids = torch.randint(3, 60, (1, 4))
        n_samples = 300

        spec_decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=5, temperature=0.8
        )
        base_decoder = BaselineDecoder(target, tokenizer, temperature=0.8)

        spec_tokens = self._collect_first_tokens(spec_decoder, input_ids, n_samples)
        base_tokens = self._collect_first_tokens(base_decoder, input_ids, n_samples)

        spec_counter = Counter(spec_tokens)
        base_counter = Counter(base_tokens)

        # Only test tokens that appeared at least 5 times in either distribution
        # (chi-squared is unreliable for very small counts)
        all_tokens = set(spec_counter.keys()) | set(base_counter.keys())
        test_tokens = [t for t in all_tokens
                       if spec_counter.get(t, 0) + base_counter.get(t, 0) >= 10]

        if len(test_tokens) < 3:
            pytest.skip("Not enough high-frequency tokens for chi-squared test")

        # Compute chi-squared statistic
        chi2 = 0.0
        for tok in test_tokens:
            observed_spec = spec_counter.get(tok, 0)
            observed_base = base_counter.get(tok, 0)
            total = observed_spec + observed_base
            expected = total / 2.0
            if expected > 0:
                chi2 += (observed_spec - expected) ** 2 / expected
                chi2 += (observed_base - expected) ** 2 / expected

        # degrees of freedom = (n_tokens - 1) * (n_methods - 1) = n_tokens - 1
        df = len(test_tokens) - 1

        # For a rough check: chi2 / df should be around 1 if distributions match
        # A value > 3 would be highly suspicious
        chi2_per_df = chi2 / df if df > 0 else 0
        assert chi2_per_df < 3.0, (
            f"Chi-squared per df too high: {chi2_per_df:.2f} (chi2={chi2:.1f}, df={df})\n"
            f"Distributions may differ significantly."
        )


class TestGreedyExactMatch:
    """At temperature=0, speculative decoding MUST produce byte-identical output."""

    def test_greedy_exact_short(self, models_and_tokenizer):
        target, draft, tokenizer = models_and_tokenizer
        input_ids = torch.randint(3, 60, (1, 5))

        set_seed(42)
        spec = SpeculativeDecoder(target, draft, tokenizer, K=5, temperature=0.0)
        spec_result = spec.generate(input_ids, max_new_tokens=20)

        set_seed(42)
        base = BaselineDecoder(target, tokenizer, temperature=0.0)
        base_result = base.generate(input_ids, max_new_tokens=20)

        min_len = min(len(spec_result.token_ids), len(base_result.token_ids))
        assert spec_result.token_ids[:min_len] == base_result.token_ids[:min_len]

    def test_greedy_exact_long(self, models_and_tokenizer):
        """Longer generation to catch subtle cache bugs."""
        target, draft, tokenizer = models_and_tokenizer
        input_ids = torch.randint(3, 60, (1, 8))

        set_seed(123)
        spec = SpeculativeDecoder(target, draft, tokenizer, K=7, temperature=0.0)
        spec_result = spec.generate(input_ids, max_new_tokens=100)

        set_seed(123)
        base = BaselineDecoder(target, tokenizer, temperature=0.0)
        base_result = base.generate(input_ids, max_new_tokens=100)

        min_len = min(len(spec_result.token_ids), len(base_result.token_ids))
        assert min_len > 0, "Both decoders should generate at least 1 token"
        assert spec_result.token_ids[:min_len] == base_result.token_ids[:min_len], (
            f"Greedy mismatch at length {min_len}!\n"
            f"Spec: {spec_result.token_ids[:20]}...\n"
            f"Base: {base_result.token_ids[:20]}..."
        )

    def test_greedy_different_K_all_match(self, models_and_tokenizer):
        """Different K values should all produce the same greedy output."""
        target, draft, tokenizer = models_and_tokenizer
        input_ids = torch.randint(3, 60, (1, 5))

        set_seed(42)
        base = BaselineDecoder(target, tokenizer, temperature=0.0)
        base_result = base.generate(input_ids, max_new_tokens=30)

        for K in [1, 2, 3, 5, 8, 12]:
            set_seed(42)
            spec = SpeculativeDecoder(target, draft, tokenizer, K=K, temperature=0.0)
            spec_result = spec.generate(input_ids, max_new_tokens=30)

            min_len = min(len(spec_result.token_ids), len(base_result.token_ids))
            assert spec_result.token_ids[:min_len] == base_result.token_ids[:min_len], (
                f"Greedy mismatch with K={K}!"
            )


class TestAdaptiveDecoder:
    """Tests for the adaptive K speculative decoder."""

    def test_generates_tokens(self, models_and_tokenizer):
        target, draft, tokenizer = models_and_tokenizer
        decoder = AdaptiveSpeculativeDecoder(
            target, draft, tokenizer, initial_K=3, temperature=1.0
        )
        input_ids = torch.randint(3, 60, (1, 5))
        result = decoder.generate(input_ids, max_new_tokens=20)
        assert result.n_generated_tokens > 0

    def test_k_history_recorded(self, models_and_tokenizer):
        target, draft, tokenizer = models_and_tokenizer
        decoder = AdaptiveSpeculativeDecoder(
            target, draft, tokenizer, initial_K=3, min_K=1, max_K=10, temperature=1.0
        )
        input_ids = torch.randint(3, 60, (1, 5))
        result = decoder.generate(input_ids, max_new_tokens=30)
        assert hasattr(result, 'k_history')
        assert len(result.k_history) > 0

    def test_k_adapts_to_high_acceptance(self, models_and_tokenizer):
        """When using the same model as draft and target, K should increase."""
        target, _, tokenizer = models_and_tokenizer
        # Use target as its own draft — acceptance should be ~100%
        decoder = AdaptiveSpeculativeDecoder(
            target, target, tokenizer, initial_K=2, min_K=1, max_K=10,
            temperature=0.0, high_threshold=0.7,
        )
        input_ids = torch.randint(3, 60, (1, 5))
        set_seed(42)
        result = decoder.generate(input_ids, max_new_tokens=50)

        if len(result.k_history) > 3:
            # K should have increased from initial value
            assert max(result.k_history) > 2, (
                f"K should have increased with perfect acceptance. History: {result.k_history}"
            )

    def test_greedy_matches_baseline(self, models_and_tokenizer):
        """Adaptive decoder at temp=0 should match baseline."""
        target, draft, tokenizer = models_and_tokenizer
        input_ids = torch.randint(3, 60, (1, 5))

        set_seed(42)
        adaptive = AdaptiveSpeculativeDecoder(
            target, draft, tokenizer, initial_K=3, temperature=0.0
        )
        adaptive_result = adaptive.generate(input_ids, max_new_tokens=20)

        set_seed(42)
        base = BaselineDecoder(target, tokenizer, temperature=0.0)
        base_result = base.generate(input_ids, max_new_tokens=20)

        min_len = min(len(adaptive_result.token_ids), len(base_result.token_ids))
        assert adaptive_result.token_ids[:min_len] == base_result.token_ids[:min_len]


class TestStress:
    """Stress tests for edge cases and robustness."""

    def test_long_generation(self, models_and_tokenizer):
        """Generate a long sequence without crashing."""
        target, draft, tokenizer = models_and_tokenizer
        decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=5, temperature=0.8
        )
        input_ids = torch.randint(3, 60, (1, 4))
        set_seed(42)
        result = decoder.generate(input_ids, max_new_tokens=500)
        assert result.n_generated_tokens > 0
        assert result.wall_clock_seconds > 0

    def test_k_equals_one(self, models_and_tokenizer):
        """K=1 is an edge case — only one draft token per step."""
        target, draft, tokenizer = models_and_tokenizer
        decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=1, temperature=1.0
        )
        input_ids = torch.randint(3, 60, (1, 4))
        set_seed(42)
        result = decoder.generate(input_ids, max_new_tokens=20)
        assert result.n_generated_tokens > 0

    def test_large_k(self, models_and_tokenizer):
        """Large K — most tokens will be rejected."""
        target, draft, tokenizer = models_and_tokenizer
        decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=20, temperature=1.0
        )
        input_ids = torch.randint(3, 60, (1, 4))
        set_seed(42)
        result = decoder.generate(input_ids, max_new_tokens=30)
        assert result.n_generated_tokens > 0

    def test_max_new_tokens_one(self, models_and_tokenizer):
        """Generate exactly 1 token."""
        target, draft, tokenizer = models_and_tokenizer
        decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=5, temperature=1.0
        )
        input_ids = torch.randint(3, 60, (1, 4))
        set_seed(42)
        result = decoder.generate(input_ids, max_new_tokens=1)
        assert result.n_generated_tokens >= 1

    def test_repeated_generation_no_memory_leak(self, models_and_tokenizer):
        """Run many generations and check that memory doesn't grow unbounded.

        On CPU this checks Python object/tensor leaks. On GPU it would
        also catch VRAM leaks.
        """
        target, draft, tokenizer = models_and_tokenizer
        decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=3, temperature=0.8
        )
        input_ids = torch.randint(3, 60, (1, 4))

        # Warmup
        for i in range(3):
            set_seed(i)
            decoder.generate(input_ids, max_new_tokens=10)

        gc.collect()
        # Count tensors before
        tensors_before = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])

        # Run 50 generations
        for i in range(50):
            set_seed(i + 100)
            decoder.generate(input_ids, max_new_tokens=15)

        gc.collect()
        tensors_after = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])

        # Allow some slack — tensor count shouldn't grow proportionally to n_runs
        tensor_growth = tensors_after - tensors_before
        assert tensor_growth < 100, (
            f"Potential memory leak: tensor count grew by {tensor_growth} "
            f"over 50 generations (before={tensors_before}, after={tensors_after})"
        )
