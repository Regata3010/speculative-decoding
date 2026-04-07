#!/usr/bin/env python3
"""Quantization precision experiment.

Tests all combinations of draft/target quantization to find the sweet spot
for speculative decoding. The hypothesis: INT8 target (slow baseline) +
FP16 draft (best predictions) maximizes speedup.

Matrix:
  Draft:  FP16, INT8, INT4
  Target: INT8, INT4
  = 6 combinations

All runs use K=5, temperature=0.0, Qwen 72B + 7B on H200.
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.model_configs import get_model_pair
from src.model_loader import load_models
from src.speculative_decoder import SpeculativeDecoder
from src.baseline_decoder import BaselineDecoder
from src.utils import set_seed, gpu_memory_stats
from benchmarks.prompts import get_sweep_subset


def run_config(target_id, draft_id, target_quant, draft_quant, K, prompts, n_repeats, seed):
    """Run one quantization config and return summary."""
    print(f"\n  Loading: target={target_quant or 'fp16'}, draft={draft_quant or 'fp16'}...")

    target_model, draft_model, tokenizer = load_models(
        target_id, draft_id,
        quantize_target=target_quant,
        quantize_draft=draft_quant,
    )

    mem = gpu_memory_stats()
    vram = mem.get("allocated_gb", 0) if mem else 0
    print(f"  VRAM: {vram:.1f} GB")

    speculative = SpeculativeDecoder(
        target_model, draft_model, tokenizer,
        K=K, temperature=0.0, top_p=1.0,
    )
    baseline = BaselineDecoder(
        target_model, tokenizer, temperature=0.0, top_p=1.0,
    )

    # Warmup
    set_seed(seed)
    input_ids = tokenizer.encode(prompts[0].prompt_text, return_tensors="pt")
    speculative.generate(input_ids, max_new_tokens=16)
    baseline.generate(input_ids, max_new_tokens=16)

    # Benchmark
    spec_tps_list = []
    base_tps_list = []
    accept_list = []
    tok_per_call_list = []

    for prompt in prompts:
        for i in range(n_repeats):
            set_seed(seed + i)
            input_ids = tokenizer.encode(prompt.prompt_text, return_tensors="pt")

            spec_result = speculative.generate(
                input_ids, max_new_tokens=prompt.max_new_tokens, profile=True
            )

            set_seed(seed + i)
            base_result = baseline.generate(
                input_ids, max_new_tokens=prompt.max_new_tokens
            )

            spec_tps_list.append(spec_result.tokens_per_second)
            base_tps_list.append(base_result.tokens_per_second)
            if spec_result.n_total_draft_tokens > 0:
                accept_list.append(spec_result.acceptance_rate)
            tok_per_call_list.append(spec_result.tokens_per_target_call)

    import statistics
    spec_tps = statistics.mean(spec_tps_list)
    base_tps = statistics.mean(base_tps_list)
    acceptance = statistics.mean(accept_list) if accept_list else 0
    tok_per_call = statistics.mean(tok_per_call_list)
    speedup = spec_tps / base_tps if base_tps > 0 else 0

    # Get profiling from last run
    prof_data = {}
    if spec_result.profiling:
        prof_data = spec_result.profiling.as_dict()

    result = {
        "target_quant": target_quant or "fp16",
        "draft_quant": draft_quant or "fp16",
        "spec_tps": round(spec_tps, 1),
        "base_tps": round(base_tps, 1),
        "speedup": round(speedup, 2),
        "acceptance_rate": round(acceptance, 3),
        "tokens_per_target_call": round(tok_per_call, 2),
        "vram_gb": round(vram, 1),
        "profiling": prof_data,
    }

    print(f"  Result: {spec_tps:.1f} tok/s, baseline {base_tps:.1f}, "
          f"speedup {speedup:.2f}x, accept {acceptance:.1%}")

    # Free memory
    del target_model, draft_model, tokenizer, speculative, baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Quantization precision experiment")
    parser.add_argument("--model-pair", default="qwen-72b", help="Model pair to test")
    parser.add_argument("--K", type=int, default=5, help="Draft tokens per step")
    parser.add_argument("--n-repeats", type=int, default=2, help="Repeats per prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/quantization_experiment")
    args = parser.parse_args()

    pair = get_model_pair(args.model_pair)
    prompts = get_sweep_subset()  # 6 diverse prompts

    configs = [
        # (target_quant, draft_quant)
        (None, "8bit"),      # FP16 target won't fit for 72B, skip
        ("8bit", None),      # INT8 target + FP16 draft (expected best)
        ("8bit", "8bit"),    # INT8 target + INT8 draft
        ("8bit", "4bit"),    # INT8 target + INT4 draft
        ("4bit", None),      # INT4 target + FP16 draft
        ("4bit", "8bit"),    # INT4 target + INT8 draft
        ("4bit", "4bit"),    # INT4 target + INT4 draft
    ]

    # Filter out FP16 target for 70B+ models (doesn't fit on single GPU)
    if "72b" in args.model_pair or "70b" in args.model_pair:
        configs = [c for c in configs if c[0] is not None]

    print("=" * 65)
    print("QUANTIZATION PRECISION EXPERIMENT")
    print("=" * 65)
    print(f"Model pair: {pair.description}")
    print(f"Target: {pair.target_id}")
    print(f"Draft:  {pair.draft_id}")
    print(f"K={args.K}, {len(prompts)} prompts, {args.n_repeats} repeats each")
    print(f"Configs to test: {len(configs)}")
    print("=" * 65)

    all_results = []
    for target_q, draft_q in configs:
        label = f"target={target_q or 'fp16'}, draft={draft_q or 'fp16'}"
        print(f"\n>>> Config: {label} <<<")
        try:
            result = run_config(
                pair.target_id, pair.draft_id,
                target_q, draft_q,
                args.K, prompts, args.n_repeats, args.seed,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({
                "target_quant": target_q or "fp16",
                "draft_quant": draft_q or "fp16",
                "error": str(e),
            })

    # Print final table
    print("\n" + "=" * 85)
    print("RESULTS TABLE")
    print("=" * 85)
    print(f"{'Draft':>8s} {'Target':>8s} {'Accept':>10s} {'Spec tok/s':>12s} {'Base tok/s':>12s} {'Speedup':>10s} {'VRAM':>8s}")
    print("-" * 85)

    for r in all_results:
        if "error" in r:
            print(f"{r['draft_quant']:>8s} {r['target_quant']:>8s} {'FAILED':>10s}")
            continue
        print(f"{r['draft_quant']:>8s} {r['target_quant']:>8s} "
              f"{r['acceptance_rate']:>9.1%} {r['spec_tps']:>12.1f} "
              f"{r['base_tps']:>12.1f} {r['speedup']:>9.2f}x {r['vram_gb']:>7.1f}G")

    print("=" * 85)

    # Sort by speedup
    valid = [r for r in all_results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["speedup"])
        print(f"\nBest config: draft={best['draft_quant']}, target={best['target_quant']} "
              f"→ {best['speedup']:.2f}x speedup")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"quant_experiment_{timestamp}.json")
    output = {
        "timestamp": datetime.now().isoformat(),
        "model_pair": args.model_pair,
        "target_id": pair.target_id,
        "draft_id": pair.draft_id,
        "K": args.K,
        "n_repeats": args.n_repeats,
        "n_prompts": len(prompts),
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
