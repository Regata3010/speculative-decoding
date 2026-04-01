#!/usr/bin/env python3
"""Run 3 — The money run.

Targets the configs most likely to show real speedup:
  1. Qwen-72B + 7B draft (large target, open access)
  2. Llama-70B + 8B draft (large target, gated)
  3. With and without torch.compile on draft
  4. Profiling on all runs

These are the runs that go on the resume.
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.model_configs import get_model_pair, list_model_pairs
from src.model_loader import load_models
from src.utils import gpu_memory_stats
from benchmarks.runner import run_benchmark


def run_single(pair_name, K, temperature, compile_draft, output_dir, n_repeats=3, quantize=None):
    pair = get_model_pair(pair_name)
    print(f"\n{'='*60}")
    print(f"  {pair_name}: {pair.description}")
    print(f"  K={K}, temp={temperature}, compile_draft={compile_draft}, quantize={quantize}")
    print(f"{'='*60}")

    target_model, draft_model, tokenizer = load_models(
        pair.target_id, pair.draft_id,
        compile_draft_only=compile_draft,
        quantize_target=quantize,
    )

    mem = gpu_memory_stats()
    if mem:
        print(f"  GPU memory: {mem['allocated_gb']:.1f} / {mem['reserved_gb']:.1f} GB")

    run_benchmark(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=K,
        temperature=temperature,
        top_p=1.0,
        n_repeats=n_repeats,
        n_warmup=3,
        seed=42,
        output_dir=output_dir,
    )

    del target_model, draft_model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run 3 — large target models")
    parser.add_argument("--output-dir", default="results/run3", help="Output directory")
    parser.add_argument("--n-repeats", type=int, default=3, help="Repeats per prompt")
    parser.add_argument("--skip-70b", action="store_true", help="Skip 70B+ models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  RUN 3 — THE MONEY RUN")
    print("=" * 60)

    # --- Small target models with torch.compile (improvement over Run 2) ---
    print("\n\n>>> PHASE A: Small targets + torch.compile draft <<<")

    run_single("llama", K=5, temperature=0.0, compile_draft=True,
               output_dir=os.path.join(args.output_dir, "llama-compiled"),
               n_repeats=args.n_repeats)

    run_single("qwen", K=3, temperature=0.0, compile_draft=True,
               output_dir=os.path.join(args.output_dir, "qwen-compiled"),
               n_repeats=args.n_repeats)

    if not args.skip_70b:
        # --- Large target models with 4-bit quantization (fits on 80GB A100) ---
        print("\n\n>>> PHASE B: Large targets + 4-bit quant (the resume numbers) <<<")

        run_single("qwen-72b", K=5, temperature=0.0, compile_draft=False,
                   output_dir=os.path.join(args.output_dir, "qwen-72b-4bit"),
                   n_repeats=args.n_repeats, quantize="4bit")

        run_single("qwen-72b", K=5, temperature=0.0, compile_draft=True,
                   output_dir=os.path.join(args.output_dir, "qwen-72b-4bit-compiled"),
                   n_repeats=args.n_repeats, quantize="4bit")

    print("\n" + "=" * 60)
    print("  RUN 3 COMPLETE")
    print(f"  Results: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
