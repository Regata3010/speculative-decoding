#!/usr/bin/env python3
"""CLI entry point for running the full benchmark suite."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_configs import get_model_pair, list_model_pairs
from src.model_loader import load_models
from src.utils import gpu_memory_stats
from benchmarks.runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run speculative decoding benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available model pairs:\n{list_model_pairs()}",
    )
    parser.add_argument("--model-pair", default=None, help="Preset model pair: llama, qwen")
    parser.add_argument("--target-model", default=None, help="Target model ID (overrides --model-pair)")
    parser.add_argument("--draft-model", default=None, help="Draft model ID (overrides --model-pair)")
    parser.add_argument("--K", type=int, default=5, help="Number of draft tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling threshold")
    parser.add_argument("--n-repeats", type=int, default=3, help="Repeats per prompt")
    parser.add_argument("--n-warmup", type=int, default=3, help="Warmup prompts to discard")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compile", action="store_true", help="torch.compile() both models")
    parser.add_argument("--compile-draft", action="store_true", help="torch.compile() draft model only (recommended)")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None, help="Quantize target model (for 70B+)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

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
        target_id, draft_id,
        compile_models=args.compile,
        compile_draft_only=args.compile_draft,
        quantize_target=args.quantize,
    )

    mem = gpu_memory_stats()
    if mem:
        print(f"  GPU memory after loading: {mem['allocated_gb']:.1f} GB")

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
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
