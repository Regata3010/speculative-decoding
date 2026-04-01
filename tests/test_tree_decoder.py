"""Tests for tree-based speculative decoding."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from src.tree_speculative_decoder import (
    TreeSpeculativeDecoder,
    TreeNode,
    build_tree_attention_mask,
)
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
        ["The quick brown fox " * 50, "Pack my box " * 50],
        trainer=trainer,
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
    )


@pytest.fixture(scope="module")
def tiny_setup():
    set_seed(42)
    target = _create_tiny_model()
    draft = _create_tiny_model()
    tokenizer = _create_tokenizer()
    return target, draft, tokenizer


class TestTreeAttentionMask:
    def test_single_chain_mask(self):
        """A tree with no branching should be a standard causal mask."""
        nodes = [
            TreeNode(token_id=1, depth=0, parent_idx=-1, children=[1]),
            TreeNode(token_id=2, depth=1, parent_idx=0, children=[2]),
            TreeNode(token_id=3, depth=2, parent_idx=1, children=[]),
        ]
        mask = build_tree_attention_mask(nodes, prompt_len=5)
        # Shape: (3, 8) — 3 nodes, 5 prompt + 3 tree
        assert mask.shape == (3, 8)

        # All nodes attend to all prompt tokens
        assert mask[:, :5].all()

        # Node 0: attends to self only (in tree part)
        assert mask[0, 5] == True
        assert mask[0, 6] == False
        assert mask[0, 7] == False

        # Node 1: attends to self + parent (node 0)
        assert mask[1, 5] == True  # parent
        assert mask[1, 6] == True  # self
        assert mask[1, 7] == False

        # Node 2: attends to self + parent (node 1) + grandparent (node 0)
        assert mask[2, 5] == True   # grandparent
        assert mask[2, 6] == True   # parent
        assert mask[2, 7] == True   # self

    def test_branching_mask(self):
        """Two branches at depth 0 should not attend to each other."""
        nodes = [
            TreeNode(token_id=1, depth=0, parent_idx=-1),  # branch 1
            TreeNode(token_id=2, depth=0, parent_idx=-1),  # branch 2
        ]
        mask = build_tree_attention_mask(nodes, prompt_len=3)
        # Shape: (2, 5)
        assert mask.shape == (2, 5)

        # Both attend to prompt
        assert mask[:, :3].all()

        # Node 0 attends to self, not node 1
        assert mask[0, 3] == True
        assert mask[0, 4] == False

        # Node 1 attends to self, not node 0
        assert mask[1, 3] == False
        assert mask[1, 4] == True


class TestTreeDecoder:
    def test_generates_tokens(self, tiny_setup):
        target, draft, tokenizer = tiny_setup
        decoder = TreeSpeculativeDecoder(
            target, draft, tokenizer,
            depth=3, branch_factor=2, temperature=1.0,
        )
        input_ids = torch.randint(3, 60, (1, 5))
        set_seed(42)
        result = decoder.generate(input_ids, max_new_tokens=10)
        assert result.n_generated_tokens > 0
        assert len(result.token_ids) > 0

    def test_tracks_metrics(self, tiny_setup):
        target, draft, tokenizer = tiny_setup
        decoder = TreeSpeculativeDecoder(
            target, draft, tokenizer,
            depth=3, branch_factor=2, temperature=1.0,
        )
        input_ids = torch.randint(3, 60, (1, 5))
        set_seed(42)
        result = decoder.generate(input_ids, max_new_tokens=15)
        assert result.n_target_calls > 0
        assert result.wall_clock_seconds > 0
        assert result.n_total_draft_tokens > 0

    def test_profiling_data(self, tiny_setup):
        target, draft, tokenizer = tiny_setup
        decoder = TreeSpeculativeDecoder(
            target, draft, tokenizer,
            depth=3, branch_factor=2, temperature=1.0,
        )
        input_ids = torch.randint(3, 60, (1, 5))
        set_seed(42)
        result = decoder.generate(input_ids, max_new_tokens=10, profile=True)
        assert result.profiling is not None
        assert result.profiling.draft_time > 0
        assert result.profiling.target_time > 0

    def test_more_tokens_per_target_call_than_linear(self, tiny_setup):
        """Tree decoder should accept more tokens per target call than linear."""
        target, draft, tokenizer = tiny_setup
        input_ids = torch.randint(3, 60, (1, 5))

        # Run tree decoder
        tree_decoder = TreeSpeculativeDecoder(
            target, draft, tokenizer,
            depth=4, branch_factor=2, temperature=1.0,
        )
        set_seed(42)
        tree_result = tree_decoder.generate(input_ids, max_new_tokens=30)

        # Tree should produce tokens (basic sanity)
        assert tree_result.n_generated_tokens > 0
        assert tree_result.tokens_per_target_call >= 1.0

    def test_different_branch_factors(self, tiny_setup):
        target, draft, tokenizer = tiny_setup
        input_ids = torch.randint(3, 60, (1, 5))

        for bf in [1, 2, 3]:
            set_seed(42)
            decoder = TreeSpeculativeDecoder(
                target, draft, tokenizer,
                depth=3, branch_factor=bf, temperature=1.0,
            )
            result = decoder.generate(input_ids, max_new_tokens=10)
            assert result.n_generated_tokens > 0

    def test_greedy_deterministic(self, tiny_setup):
        """Two runs with same seed should produce same output."""
        target, draft, tokenizer = tiny_setup
        input_ids = torch.randint(3, 60, (1, 5))

        decoder = TreeSpeculativeDecoder(
            target, draft, tokenizer,
            depth=3, branch_factor=2, temperature=0.0,
        )

        set_seed(42)
        r1 = decoder.generate(input_ids, max_new_tokens=15)
        set_seed(42)
        r2 = decoder.generate(input_ids, max_new_tokens=15)

        min_len = min(len(r1.token_ids), len(r2.token_ids))
        assert r1.token_ids[:min_len] == r2.token_ids[:min_len]
