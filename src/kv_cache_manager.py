"""KV cache management for speculative decoding.

Handles cache creation, rollback after rejection, and length tracking
for both draft and target models.
"""

from transformers import DynamicCache


def create_cache() -> DynamicCache:
    """Create a new empty KV cache."""
    return DynamicCache()


def get_cache_length(cache: DynamicCache) -> int:
    """Get the current sequence length stored in the cache.

    Returns 0 for an empty cache.
    """
    if cache is None:
        return 0
    seq_length = cache.get_seq_length()
    return seq_length if seq_length is not None else 0


def rollback_cache(cache: DynamicCache, max_length: int) -> DynamicCache:
    """Roll back a KV cache to a specified sequence length.

    After draft token rejection at position i, both the draft and target
    caches need to be cropped to discard the rejected positions.

    Args:
        cache: The DynamicCache to roll back.
        max_length: The sequence length to crop to.

    Returns:
        The cropped cache (modified in-place and returned).
    """
    if cache is None:
        return cache

    current_length = get_cache_length(cache)
    if current_length > max_length:
        cache.crop(max_length)

    return cache
