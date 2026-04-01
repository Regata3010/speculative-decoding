#!/usr/bin/env python3
"""CLI entry point for hyperparameter sweep."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_configs import get_model_pair, list_model_pairs
from src.model_loader import load_models
from src.utils import gpu_memory_stats
from benchmarks.sweep import run_sweep


def main():
    parser = argparse.ArgumentParser(
        description="Run speculative decoding hyperparameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available model pairs:\n{list_model_pairs()}",
    )
    parser.add_argument("--model-pair", default=None, help="Preset model pair: llama, qwen")
    parser.add_argument("--target-model", default=None, help="Target model ID (overrides --model-pair)")
    parser.add_argument("--draft-model", default=None, help="Draft model ID (overrides --model-pair)")
    parser.add_argument("--K-values", type=str, default=None, help="Comma-separated K values (e.g., 1,3,5,7)")
    parser.add_argument("--temperatures", type=str, default=None, help="Comma-separated temperatures")
    parser.add_argument("--top-p-values", type=str, default=None, help="Comma-separated top-p values")
    parser.add_argument("--n-repeats", type=int, default=2, help="Repeats per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for models")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    K_values = [int(x) for x in args.K_values.split(",")] if args.K_values else None
    temperatures = [float(x) for x in args.temperatures.split(",")] if args.temperatures else None
    top_p_values = [float(x) for x in args.top_p_values.split(",")] if args.top_p_values else None

    # Resolve model IDs
    if args.target_model and args.draft_model:
        target_id = args.target_model
        draft_id = args.draft_model
    elif args.model_pair:
        pair = get_model_pair(args.model_pair)
        target_id = args.target_model or pair.target_id
        draft_id = args.draft_model or pair.draft_id
    else:
        pair = get_model_pair("qwen")
        target_id = pair.target_id
        draft_id = pair.draft_id

    print(f"Loading models...")
    print(f"  Target: {target_id}")
    print(f"  Draft:  {draft_id}")
    target_model, draft_model, tokenizer = load_models(
        target_id, draft_id, compile_models=args.compile
    )

    mem = gpu_memory_stats()
    if mem:
        print(f"GPU memory after loading: {mem['allocated_gb']:.1f} GB")

    run_sweep(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K_values=K_values,
        temperatures=temperatures,
        top_p_values=top_p_values,
        n_repeats=args.n_repeats,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
