#!/usr/bin/env python3
"""Generate plots from Run 2 results with hardcoded paths."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_plots import generate_all_plots, plot_profiling_breakdown

# Run 2 Qwen sweep
qwen_sweep = "run2 benhmarks/qwen/benchmark_20260330_233137.json"  # fallback
llama_bench = "run2 benhmarks/benchmark_20260331_034558.json"
qwen_1_5b_bench = "run2 benhmarks/benchmark_20260331_031631.json"

# Find sweep CSVs in run1results (those are the sweep files)
run1_qwen_sweep = "run1results/sweep_20260331_001541.csv"
run1_llama_sweep = "run1results/sweep_20260331_011518.csv"

os.makedirs("plots/run2", exist_ok=True)

# Generate profiling breakdown plots from Run 2 benchmarks (they have profiling data)
print("=== Generating Run 2 profiling plots ===")
if os.path.exists(qwen_1_5b_bench):
    print(f"\nQwen 1.5B draft profiling:")
    plot_profiling_breakdown(qwen_1_5b_bench, "plots/run2")
    # Rename to be specific
    if os.path.exists("plots/run2/profiling_breakdown.png"):
        os.rename("plots/run2/profiling_breakdown.png", "plots/run2/profiling_qwen_1.5b.png")
        print("  Saved: profiling_qwen_1.5b.png")

if os.path.exists(llama_bench):
    print(f"\nLlama 1B draft profiling:")
    plot_profiling_breakdown(llama_bench, "plots/run2")
    if os.path.exists("plots/run2/profiling_breakdown.png"):
        os.rename("plots/run2/profiling_breakdown.png", "plots/run2/profiling_llama_1b.png")
        print("  Saved: profiling_llama_1b.png")

print("\n=== Done ===")
print("Plots saved to plots/run2/")
