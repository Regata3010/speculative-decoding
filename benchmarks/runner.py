"""Benchmark runner — orchestrates speculative vs baseline comparisons."""

import json
import os
from datetime import datetime

import torch
from tqdm import tqdm

from benchmarks.metrics import aggregate_results, results_to_dict
from benchmarks.prompts import PROMPTS, BenchmarkPrompt
from src.baseline_decoder import BaselineDecoder
from src.hf_assisted_decoder import HFAssistedDecoder
from src.speculative_decoder import SpeculativeDecoder
from src.tree_speculative_decoder import TreeSpeculativeDecoder
from src.utils import GenerationResult, gpu_memory_stats, set_seed


def run_single_prompt(
    decoder,
    tokenizer,
    prompt: BenchmarkPrompt,
    n_repeats: int = 3,
    seed: int = 42,
) -> list[GenerationResult]:
    """Run a single prompt multiple times and return results."""
    device = next(iter(decoder.target_model.parameters())).device if hasattr(decoder, 'target_model') else next(iter(decoder.target_model.parameters())).device
    results = []
    for i in range(n_repeats):
        set_seed(seed + i)
        input_ids = tokenizer.encode(prompt.prompt_text, return_tensors="pt")
        result = decoder.generate(input_ids, max_new_tokens=prompt.max_new_tokens)
        results.append(result)
    return results


def run_benchmark(
    target_model,
    draft_model,
    tokenizer,
    prompts: list[BenchmarkPrompt] | None = None,
    K: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n_repeats: int = 3,
    n_warmup: int = 3,
    seed: int = 42,
    output_dir: str = "results",
) -> dict:
    """Run full benchmark comparing speculative vs baseline decoding.

    Args:
        target_model: The large target model.
        draft_model: The small draft model.
        tokenizer: Shared tokenizer.
        prompts: List of prompts to benchmark. Defaults to full suite.
        K: Number of draft tokens per speculative step.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        n_repeats: Number of times to run each prompt.
        n_warmup: Number of warmup prompts to discard.
        seed: Random seed.
        output_dir: Directory to save results.

    Returns:
        Dict with aggregated results and per-prompt details.
    """
    if prompts is None:
        prompts = PROMPTS

    speculative = SpeculativeDecoder(
        target_model, draft_model, tokenizer, K=K,
        temperature=temperature, top_p=top_p,
    )
    # Tree speculative decoder disabled in default benchmarks — cache deepcopy
    # overhead (73% of time) negates tree benefits in pure Python. Documented
    # as an experimental finding. Enable manually for research:
    # tree_speculative = TreeSpeculativeDecoder(
    #     target_model, draft_model, tokenizer,
    #     depth=K, branch_factor=2,
    #     temperature=temperature, top_p=top_p,
    # )
    baseline = BaselineDecoder(
        target_model, tokenizer,
        temperature=temperature, top_p=top_p,
    )
    hf_assisted = HFAssistedDecoder(
        target_model, draft_model, tokenizer,
        temperature=temperature, top_p=top_p,
    )

    # Warmup
    print(f"Warming up with {n_warmup} prompts...")
    for prompt in prompts[:n_warmup]:
        set_seed(seed)
        input_ids = tokenizer.encode(prompt.prompt_text, return_tensors="pt")
        speculative.generate(input_ids, max_new_tokens=min(prompt.max_new_tokens, 32))
        baseline.generate(input_ids, max_new_tokens=min(prompt.max_new_tokens, 32))
        hf_assisted.generate(input_ids, max_new_tokens=min(prompt.max_new_tokens, 32))

    # Benchmark
    all_spec_results = []
    all_base_results = []
    all_hf_results = []
    per_prompt_data = []

    print(f"\nBenchmarking {len(prompts)} prompts x {n_repeats} repeats...")
    for prompt in tqdm(prompts, desc="Prompts"):
        spec_results = []
        base_results = []
        hf_results = []

        for i in range(n_repeats):
            set_seed(seed + i)
            input_ids = tokenizer.encode(prompt.prompt_text, return_tensors="pt")

            # Run linear speculative decoder (with profiling)
            spec_result = speculative.generate(
                input_ids, max_new_tokens=prompt.max_new_tokens, profile=True
            )
            spec_results.append(spec_result)

            # Run baseline with same seed
            set_seed(seed + i)
            base_result = baseline.generate(
                input_ids, max_new_tokens=prompt.max_new_tokens
            )
            base_results.append(base_result)

            # Run HF assisted generation with same seed
            set_seed(seed + i)
            hf_result = hf_assisted.generate(
                input_ids, max_new_tokens=prompt.max_new_tokens
            )
            hf_results.append(hf_result)

        all_spec_results.extend(spec_results)
        all_base_results.extend(base_results)
        all_hf_results.extend(hf_results)

        per_prompt_data.append({
            "prompt_name": prompt.name,
            "category": prompt.category,
            "max_new_tokens": prompt.max_new_tokens,
            "speculative": [results_to_dict(r, prompt.name) for r in spec_results],
            "baseline": [results_to_dict(r, prompt.name) for r in base_results],
            "hf_assisted": [results_to_dict(r, prompt.name) for r in hf_results],
        })

    # Aggregate
    spec_agg = aggregate_results(all_spec_results, all_base_results)
    base_agg = aggregate_results(all_base_results)
    hf_agg = aggregate_results(all_hf_results, all_base_results)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "K": K,
            "temperature": temperature,
            "top_p": top_p,
            "n_repeats": n_repeats,
            "n_warmup": n_warmup,
            "n_prompts": len(prompts),
        },
        "gpu_memory": gpu_memory_stats(),
        "speculative_aggregate": {
            "acceptance_rate_mean": spec_agg.acceptance_rate_mean,
            "acceptance_rate_std": spec_agg.acceptance_rate_std,
            "tokens_per_sec_mean": spec_agg.tokens_per_sec_mean,
            "tokens_per_sec_std": spec_agg.tokens_per_sec_std,
            "tokens_per_sec_median": spec_agg.tokens_per_sec_median,
            "speedup_mean": spec_agg.speedup_mean,
            "speedup_std": spec_agg.speedup_std,
            "tokens_per_target_call_mean": spec_agg.tokens_per_target_call_mean,
        },
        "baseline_aggregate": {
            "tokens_per_sec_mean": base_agg.tokens_per_sec_mean,
            "tokens_per_sec_std": base_agg.tokens_per_sec_std,
            "tokens_per_sec_median": base_agg.tokens_per_sec_median,
        },
        "hf_assisted_aggregate": {
            "tokens_per_sec_mean": hf_agg.tokens_per_sec_mean,
            "tokens_per_sec_std": hf_agg.tokens_per_sec_std,
            "tokens_per_sec_median": hf_agg.tokens_per_sec_median,
            "speedup_mean": hf_agg.speedup_mean,
            "speedup_std": hf_agg.speedup_std,
        },
        "per_prompt": per_prompt_data,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Speculative Decoding (K={K}):")
    print(f"  Acceptance Rate:     {spec_agg.acceptance_rate_mean:.1%} +/- {spec_agg.acceptance_rate_std:.1%}")
    print(f"  Tokens/sec:          {spec_agg.tokens_per_sec_mean:.1f} +/- {spec_agg.tokens_per_sec_std:.1f}")
    print(f"  Tokens/target call:  {spec_agg.tokens_per_target_call_mean:.2f}")
    print(f"  Speedup:             {spec_agg.speedup_mean:.2f}x")
    print(f"\nBaseline (autoregressive):")
    print(f"  Tokens/sec:          {base_agg.tokens_per_sec_mean:.1f} +/- {base_agg.tokens_per_sec_std:.1f}")
    print(f"\nHF Assisted Generation:")
    print(f"  Tokens/sec:          {hf_agg.tokens_per_sec_mean:.1f} +/- {hf_agg.tokens_per_sec_std:.1f}")
    print(f"  Speedup:             {hf_agg.speedup_mean:.2f}x")
    print("=" * 60)

    return output
