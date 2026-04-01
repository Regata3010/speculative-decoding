"""Metrics computation and aggregation for benchmark results."""

from dataclasses import dataclass

import numpy as np

from src.utils import GenerationResult


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple benchmark runs."""

    acceptance_rate_mean: float
    acceptance_rate_std: float
    tokens_per_sec_mean: float
    tokens_per_sec_std: float
    tokens_per_sec_median: float
    tokens_per_sec_p5: float
    tokens_per_sec_p95: float
    speedup_mean: float
    speedup_std: float
    latency_per_token_mean: float
    latency_per_token_std: float
    tokens_per_target_call_mean: float
    tokens_per_target_call_std: float
    n_runs: int


def compute_speedup(
    speculative_results: list[GenerationResult],
    baseline_results: list[GenerationResult],
) -> list[float]:
    """Compute per-prompt speedup ratios."""
    speedups = []
    for spec, base in zip(speculative_results, baseline_results):
        if spec.tokens_per_second > 0 and base.tokens_per_second > 0:
            speedups.append(spec.tokens_per_second / base.tokens_per_second)
    return speedups


def aggregate_results(
    results: list[GenerationResult],
    baseline_results: list[GenerationResult] | None = None,
) -> AggregatedMetrics:
    """Aggregate metrics across multiple GenerationResults."""
    acceptance_rates = [r.acceptance_rate for r in results if r.n_total_draft_tokens > 0]
    tps = [r.tokens_per_second for r in results if r.tokens_per_second > 0]
    latencies = [r.latency_per_token for r in results if r.latency_per_token > 0]
    tpc = [r.tokens_per_target_call for r in results if r.n_target_calls > 0]

    speedups = []
    if baseline_results:
        speedups = compute_speedup(results, baseline_results)

    def _safe_stats(values):
        if not values:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        arr = np.array(values)
        return (
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.median(arr)),
            float(np.percentile(arr, 5)),
            float(np.percentile(arr, 95)),
        )

    ar_mean, ar_std, _, _, _ = _safe_stats(acceptance_rates)
    tps_mean, tps_std, tps_median, tps_p5, tps_p95 = _safe_stats(tps)
    sp_mean, sp_std, _, _, _ = _safe_stats(speedups)
    lat_mean, lat_std, _, _, _ = _safe_stats(latencies)
    tpc_mean, tpc_std, _, _, _ = _safe_stats(tpc)

    return AggregatedMetrics(
        acceptance_rate_mean=ar_mean,
        acceptance_rate_std=ar_std,
        tokens_per_sec_mean=tps_mean,
        tokens_per_sec_std=tps_std,
        tokens_per_sec_median=tps_median,
        tokens_per_sec_p5=tps_p5,
        tokens_per_sec_p95=tps_p95,
        speedup_mean=sp_mean,
        speedup_std=sp_std,
        latency_per_token_mean=lat_mean,
        latency_per_token_std=lat_std,
        tokens_per_target_call_mean=tpc_mean,
        tokens_per_target_call_std=tpc_std,
        n_runs=len(results),
    )


def results_to_dict(result: GenerationResult, prompt_name: str = "") -> dict:
    """Convert a GenerationResult to a JSON-serializable dict."""
    d = {
        "prompt_name": prompt_name,
        "n_generated_tokens": result.n_generated_tokens,
        "n_target_calls": result.n_target_calls,
        "n_accepted_draft_tokens": result.n_accepted_draft_tokens,
        "n_total_draft_tokens": result.n_total_draft_tokens,
        "acceptance_rate": result.acceptance_rate,
        "tokens_per_second": result.tokens_per_second,
        "tokens_per_target_call": result.tokens_per_target_call,
        "latency_per_token": result.latency_per_token,
        "wall_clock_seconds": result.wall_clock_seconds,
    }
    if result.profiling is not None:
        d["profiling"] = result.profiling.as_dict()
    return d
