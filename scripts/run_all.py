#!/usr/bin/env python3
"""Run benchmarks across all model pairs.

Benchmarks each configured model pair (Qwen, Llama, etc.) with the same
settings, producing separate result files for cross-model comparison.
"""

import argparse
import gc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.model_configs import MODEL_PAIRS, list_model_pairs
from src.model_loader import load_models
from src.utils import gpu_memory_stats
from benchmarks.runner import run_benchmark
from benchmarks.sweep import run_sweep


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks across all model pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available model pairs:\n{list_model_pairs()}",
    )
    parser.add_argument(
        "--pairs", default="all",
        help="Comma-separated model pairs to run (e.g., qwen,llama) or 'all'",
    )
    parser.add_argument("--K", type=int, default=5, help="Number of draft tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling threshold")
    parser.add_argument("--n-repeats", type=int, default=3, help="Repeats per prompt")
    parser.add_argument("--n-warmup", type=int, default=3, help="Warmup prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument("--sweep", action="store_true", help="Also run hyperparameter sweep")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    # Determine which pairs to run
    if args.pairs == "all":
        pair_names = list(MODEL_PAIRS.keys())
    else:
        pair_names = [p.strip() for p in args.pairs.split(",")]

    print(f"Will benchmark {len(pair_names)} model pair(s): {', '.join(pair_names)}")
    print("=" * 60)

    for i, pair_name in enumerate(pair_names):
        pair = MODEL_PAIRS[pair_name]
        pair_output_dir = os.path.join(args.output_dir, pair_name)

        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{len(pair_names)}] {pair.name.upper()}: {pair.description}")
        print(f"{'=' * 60}")

        print(f"\nLoading {pair_name} models...")
        print(f"  Target: {pair.target_id}")
        print(f"  Draft:  {pair.draft_id}")

        target_model, draft_model, tokenizer = load_models(
            pair.target_id, pair.draft_id, compile_models=args.compile,
        )

        mem = gpu_memory_stats()
        if mem:
            print(f"  GPU memory: {mem['allocated_gb']:.1f} GB")

        # Run benchmark
        print(f"\n--- Benchmark (K={args.K}) ---")
        run_benchmark(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            K=args.K,
            temperature=args.temperature,
            top_p=args.top_p,
            n_repeats=args.n_repeats,
            n_warmup=args.n_warmup,
            seed=args.seed,
            output_dir=pair_output_dir,
        )

        # Optional sweep
        if args.sweep:
            print(f"\n--- Hyperparameter Sweep ---")
            run_sweep(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                n_repeats=min(args.n_repeats, 2),
                seed=args.seed,
                output_dir=pair_output_dir,
            )

        # Free GPU memory before loading next pair
        del target_model, draft_model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("ALL BENCHMARKS COMPLETE")
    print(f"Results saved under: {args.output_dir}/")
    for pair_name in pair_names:
        print(f"  {pair_name}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
