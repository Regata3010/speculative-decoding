"""Integration tests for the speculative decoder.

Uses small random-weight models to verify the end-to-end loop runs
correctly without requiring GPU or large model downloads.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from src.speculative_decoder import SpeculativeDecoder
from src.baseline_decoder import BaselineDecoder
from src.utils import set_seed


def _create_tiny_model(vocab_size=256, hidden_size=64, num_layers=2, num_heads=2):
    """Create a tiny random-weight LlamaForCausalLM for testing."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _create_tiny_tokenizer():
    """Create a simple tokenizer for testing."""
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    # Build a minimal BPE tokenizer
    tokenizer_model = models.BPE()
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=256,
        special_tokens=["<pad>", "<eos>", "<unk>"],
    )
    # Train on dummy text
    tokenizer.train_from_iterator(
        ["Hello world this is a test " * 10, "The quick brown fox " * 10],
        trainer=trainer,
    )

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    return wrapped


@pytest.fixture
def tiny_setup():
    """Set up tiny models and tokenizer for testing."""
    set_seed(42)
    target = _create_tiny_model()
    draft = _create_tiny_model()
    tokenizer = _create_tiny_tokenizer()
    return target, draft, tokenizer


class TestSpeculativeDecoder:
    def test_generates_tokens(self, tiny_setup):
        target, draft, tokenizer = tiny_setup
        decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=3, temperature=1.0
        )
        input_ids = torch.randint(0, 256, (1, 5))
        result = decoder.generate(input_ids, max_new_tokens=10)

        assert result.n_generated_tokens > 0
        assert len(result.token_ids) > 0
        assert result.wall_clock_seconds > 0

    def test_respects_max_new_tokens(self, tiny_setup):
        target, draft, tokenizer = tiny_setup
        decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=3, temperature=1.0
        )
        input_ids = torch.randint(0, 256, (1, 5))
        result = decoder.generate(input_ids, max_new_tokens=5)

        assert result.n_generated_tokens <= 5 + decoder.K + 1  # Allow slight overshoot from last spec step

    def test_tracks_metrics(self, tiny_setup):
        target, draft, tokenizer = tiny_setup
        decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=3, temperature=1.0
        )
        input_ids = torch.randint(0, 256, (1, 5))
        result = decoder.generate(input_ids, max_new_tokens=20)

        assert result.n_target_calls > 0
        assert result.tokens_per_second >= 0
        assert 0 <= result.acceptance_rate <= 1.0

    def test_different_K_values(self, tiny_setup):
        target, draft, tokenizer = tiny_setup
        input_ids = torch.randint(0, 256, (1, 5))

        for K in [1, 3, 5, 10]:
            set_seed(42)
            decoder = SpeculativeDecoder(
                target, draft, tokenizer, K=K, temperature=1.0
            )
            result = decoder.generate(input_ids, max_new_tokens=10)
            assert result.n_generated_tokens > 0


class TestBaselineDecoder:
    def test_generates_tokens(self, tiny_setup):
        target, _, tokenizer = tiny_setup
        decoder = BaselineDecoder(target, tokenizer, temperature=1.0)
        input_ids = torch.randint(0, 256, (1, 5))
        result = decoder.generate(input_ids, max_new_tokens=10)

        assert result.n_generated_tokens > 0
        assert result.n_target_calls > 0
        assert result.tokens_per_second >= 0

    def test_respects_max_new_tokens(self, tiny_setup):
        target, _, tokenizer = tiny_setup
        decoder = BaselineDecoder(target, tokenizer, temperature=1.0)
        input_ids = torch.randint(0, 256, (1, 5))
        result = decoder.generate(input_ids, max_new_tokens=5)

        assert result.n_generated_tokens <= 5


class TestGreedyEquivalence:
    """At temperature=0, speculative decoding should produce identical output to baseline."""

    def test_greedy_output_matches(self, tiny_setup):
        target, draft, tokenizer = tiny_setup
        input_ids = torch.randint(0, 256, (1, 5))

        set_seed(42)
        spec_decoder = SpeculativeDecoder(
            target, draft, tokenizer, K=3, temperature=0.0
        )
        spec_result = spec_decoder.generate(input_ids, max_new_tokens=15)

        set_seed(42)
        base_decoder = BaselineDecoder(target, tokenizer, temperature=0.0)
        base_result = base_decoder.generate(input_ids, max_new_tokens=15)

        # With greedy decoding and same model, outputs should be identical
        min_len = min(len(spec_result.token_ids), len(base_result.token_ids))
        assert spec_result.token_ids[:min_len] == base_result.token_ids[:min_len], (
            f"Greedy outputs differ!\n"
            f"Speculative: {spec_result.token_ids[:min_len]}\n"
            f"Baseline:    {base_result.token_ids[:min_len]}"
        )
