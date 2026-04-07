"""Utility functions for timing, seeding, and device management."""

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ProfilingData:
    """Per-phase timing breakdown for speculative decoding."""

    draft_time: float = 0.0       # Time spent in draft model forward passes
    target_time: float = 0.0      # Time spent in target model forward passes
    sampling_time: float = 0.0    # Time spent in rejection sampling
    cache_time: float = 0.0       # Time spent in cache rollback
    overhead_time: float = 0.0    # Python glue / tensor creation / everything else

    @property
    def total_time(self) -> float:
        return self.draft_time + self.target_time + self.sampling_time + self.cache_time + self.overhead_time

    def as_dict(self) -> dict:
        total = self.total_time
        if total == 0:
            return {}
        return {
            "draft_time_s": round(self.draft_time, 4),
            "target_time_s": round(self.target_time, 4),
            "sampling_time_s": round(self.sampling_time, 4),
            "cache_time_s": round(self.cache_time, 4),
            "overhead_time_s": round(self.overhead_time, 4),
            "draft_pct": round(100 * self.draft_time / total, 1),
            "target_pct": round(100 * self.target_time / total, 1),
            "sampling_pct": round(100 * self.sampling_time / total, 1),
            "cache_pct": round(100 * self.cache_time / total, 1),
            "overhead_pct": round(100 * self.overhead_time / total, 1),
        }


@dataclass
class GenerationResult:
    """Container for generation outputs and metrics."""

    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    n_target_calls: int = 0
    n_generated_tokens: int = 0
    n_accepted_draft_tokens: int = 0
    n_total_draft_tokens: int = 0
    wall_clock_seconds: float = 0.0
    profiling: ProfilingData | None = None
    # Per-position acceptance tracking: position_accepted[i] = number of times
    # position i in the draft sequence was accepted. position_proposed[i] = number
    # of times position i was proposed (reached before rejection).
    position_accepted: list[int] = field(default_factory=list)
    position_proposed: list[int] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        if self.wall_clock_seconds == 0:
            return 0.0
        return self.n_generated_tokens / self.wall_clock_seconds

    @property
    def acceptance_rate(self) -> float:
        if self.n_total_draft_tokens == 0:
            return 0.0
        return self.n_accepted_draft_tokens / self.n_total_draft_tokens

    @property
    def tokens_per_target_call(self) -> float:
        if self.n_target_calls == 0:
            return 0.0
        return self.n_generated_tokens / self.n_target_calls

    @property
    def latency_per_token(self) -> float:
        if self.n_generated_tokens == 0:
            return 0.0
        return self.wall_clock_seconds / self.n_generated_tokens

    @property
    def per_position_acceptance_rate(self) -> list[float]:
        """Acceptance rate at each draft position (0 = first draft token, K-1 = last).

        Position 0 is always reached. Position i is only reached if positions
        0..i-1 were all accepted. A declining curve shows that later positions
        are harder to predict — the draft model's accuracy degrades further
        from the last verified token.
        """
        rates = []
        for acc, prop in zip(self.position_accepted, self.position_proposed):
            rates.append(acc / prop if prop > 0 else 0.0)
        return rates


class CudaTimer:
    """Accurate GPU timing using CUDA events.

    Falls back to time.perf_counter() on CPU.
    """

    def __init__(self, device: torch.device):
        self.use_cuda = device.type == "cuda"
        self._start_event = None
        self._end_event = None
        self._start_time = None

    def start(self):
        if self.use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timer and return elapsed time in seconds."""
        if self.use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            return self._start_event.elapsed_time(self._end_event) / 1000.0
        else:
            return time.perf_counter() - self._start_time


@contextmanager
def cuda_timer(device: torch.device):
    """Context manager for timing a block of code.

    Yields a list that will contain [elapsed_seconds] after the block.
    """
    timer = CudaTimer(device)
    result = []
    timer.start()
    try:
        yield result
    finally:
        result.append(timer.stop())


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """Get the best dtype for the device. bf16 for CUDA if supported, else fp16."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def gpu_memory_stats() -> dict[str, float]:
    """Return GPU memory usage in GB. Empty dict if no CUDA."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
