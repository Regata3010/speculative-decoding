"""Tests for KV cache management."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from src.kv_cache_manager import create_cache, get_cache_length, rollback_cache


class TestCacheCreation:
    def test_create_empty_cache(self):
        cache = create_cache()
        assert cache is not None
        assert get_cache_length(cache) == 0

    def test_none_cache_length(self):
        assert get_cache_length(None) == 0


class TestCacheRollback:
    def test_rollback_none_cache(self):
        result = rollback_cache(None, 5)
        assert result is None

    def test_rollback_within_bounds(self):
        cache = create_cache()
        # Manually populate cache with dummy data to test rollback
        # DynamicCache stores (key, value) pairs per layer
        batch_size = 1
        n_heads = 4
        head_dim = 8
        seq_len = 10

        key = torch.randn(batch_size, n_heads, seq_len, head_dim)
        value = torch.randn(batch_size, n_heads, seq_len, head_dim)
        cache.update(key, value, layer_idx=0)

        assert get_cache_length(cache) == seq_len

        # Rollback to 5
        rollback_cache(cache, 5)
        assert get_cache_length(cache) == 5

    def test_rollback_no_op_when_shorter(self):
        cache = create_cache()
        key = torch.randn(1, 4, 5, 8)
        value = torch.randn(1, 4, 5, 8)
        cache.update(key, value, layer_idx=0)

        # Rollback to larger than current — should be no-op
        rollback_cache(cache, 10)
        assert get_cache_length(cache) == 5

    def test_rollback_to_zero(self):
        cache = create_cache()
        key = torch.randn(1, 4, 5, 8)
        value = torch.randn(1, 4, 5, 8)
        cache.update(key, value, layer_idx=0)

        rollback_cache(cache, 0)
        assert get_cache_length(cache) == 0

    def test_rollback_preserves_data(self):
        cache = create_cache()
        key = torch.randn(1, 4, 10, 8)
        value = torch.randn(1, 4, 10, 8)
        cache.update(key, value, layer_idx=0)

        # Store first 5 positions' data
        original_key_prefix = key[:, :, :5, :].clone()

        rollback_cache(cache, 5)

        # After rollback, the remaining data should match
        cached_key = cache.layers[0].keys
        assert torch.allclose(cached_key[:, :, :5, :], original_key_prefix)
