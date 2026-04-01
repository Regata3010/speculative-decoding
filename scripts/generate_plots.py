#!/usr/bin/env python3
"""Generate publication-quality plots from benchmark and sweep results."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "speculative": "#2196F3",
    "baseline": "#FF5722",
    "accent": "#4CAF50",
}


def plot_speedup_vs_k(df: pd.DataFrame, output_dir: str):
    """Hero figure: tokens/sec at different K values vs baseline."""
    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = df.groupby("K").agg(
        spec_tps=("spec_tokens_per_sec", "mean"),
        spec_tps_std=("spec_tokens_per_sec", "std"),
        base_tps=("base_tokens_per_sec", "mean"),
    ).reset_index()

    ax.errorbar(
        grouped["K"], grouped["spec_tps"], yerr=grouped["spec_tps_std"],
        marker="o", linewidth=2, capsize=4, color=COLORS["speculative"],
        label="Speculative Decoding",
    )
    ax.axhline(
        y=grouped["base_tps"].mean(), color=COLORS["baseline"],
        linestyle="--", linewidth=2, label="Baseline (autoregressive)",
    )

    ax.set_xlabel("K (draft tokens per step)")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Speculative Decoding Throughput vs Draft Length K")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "speedup_vs_k.png"))
    plt.close()
    print("  Generated: speedup_vs_k.png")


def plot_acceptance_rate_vs_k(df: pd.DataFrame, output_dir: str):
    """Acceptance rate vs K showing diminishing returns."""
    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = df.groupby("K").agg(
        ar_mean=("acceptance_rate", "mean"),
        ar_std=("acceptance_rate", "std"),
    ).reset_index()

    ax.errorbar(
        grouped["K"], grouped["ar_mean"], yerr=grouped["ar_std"],
        marker="s", linewidth=2, capsize=4, color=COLORS["accent"],
    )

    ax.set_xlabel("K (draft tokens per step)")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Draft Token Acceptance Rate vs K")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "acceptance_rate_vs_k.png"))
    plt.close()
    print("  Generated: acceptance_rate_vs_k.png")


def plot_acceptance_heatmap(df: pd.DataFrame, output_dir: str):
    """Heatmap of acceptance rate: K vs temperature."""
    pivot = df.groupby(["K", "temperature"])["acceptance_rate"].mean().reset_index()
    heatmap_data = pivot.pivot(index="K", columns="temperature", values="acceptance_rate")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu",
        cbar_kws={"label": "Acceptance Rate"}, ax=ax,
    )
    ax.set_title("Acceptance Rate: K vs Temperature")
    ax.set_ylabel("K (draft tokens)")
    ax.set_xlabel("Temperature")

    plt.savefig(os.path.join(output_dir, "acceptance_heatmap.png"))
    plt.close()
    print("  Generated: acceptance_heatmap.png")


def plot_per_category_speedup(df: pd.DataFrame, output_dir: str):
    """Grouped bar chart of speedup across task categories."""
    fig, ax = plt.subplots(figsize=(10, 5))

    category_speedup = df.groupby("category")["speedup"].agg(["mean", "std"]).reset_index()
    category_speedup = category_speedup.sort_values("mean", ascending=True)

    bars = ax.barh(
        category_speedup["category"], category_speedup["mean"],
        xerr=category_speedup["std"], capsize=4,
        color=COLORS["speculative"], alpha=0.8,
    )
    ax.axvline(x=1.0, color=COLORS["baseline"], linestyle="--", linewidth=1.5, label="1x (no speedup)")
    ax.set_xlabel("Speedup over Baseline")
    ax.set_title("Speculative Decoding Speedup by Task Category")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.savefig(os.path.join(output_dir, "per_category_speedup.png"))
    plt.close()
    print("  Generated: per_category_speedup.png")


def plot_tokens_per_target_call(df: pd.DataFrame, output_dir: str):
    """Tokens per target forward pass — the compression ratio."""
    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = df.groupby("K").agg(
        tpc_mean=("tokens_per_target_call", "mean"),
        tpc_std=("tokens_per_target_call", "std"),
    ).reset_index()

    ax.bar(
        grouped["K"].astype(str), grouped["tpc_mean"],
        yerr=grouped["tpc_std"], capsize=4,
        color=COLORS["speculative"], alpha=0.8,
    )
    ax.axhline(y=1.0, color=COLORS["baseline"], linestyle="--", linewidth=1.5,
               label="Baseline (1 token/call)")
    ax.set_xlabel("K (draft tokens per step)")
    ax.set_ylabel("Tokens per Target Forward Pass")
    ax.set_title("Effective Tokens per Target Model Call")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(os.path.join(output_dir, "tokens_per_target_call.png"))
    plt.close()
    print("  Generated: tokens_per_target_call.png")


def plot_speedup_distribution(df: pd.DataFrame, output_dir: str):
    """Distribution of speedup values across all runs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(df["speedup"], bins=30, color=COLORS["speculative"], alpha=0.7, edgecolor="black")
    ax.axvline(x=df["speedup"].mean(), color=COLORS["baseline"], linestyle="--",
               linewidth=2, label=f'Mean: {df["speedup"].mean():.2f}x')
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1.5, label="1x (break-even)")
    ax.set_xlabel("Speedup")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Speedup Across All Runs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "speedup_distribution.png"))
    plt.close()
    print("  Generated: speedup_distribution.png")


def plot_profiling_breakdown(benchmark_json: str, output_dir: str):
    """Stacked bar chart showing where time goes: draft, target, sampling, overhead."""
    with open(benchmark_json) as f:
        data = json.load(f)

    # Aggregate profiling data across all prompts
    phases = {"Draft Model": 0, "Target Model": 0, "Sampling": 0, "Cache": 0, "Overhead": 0}
    count = 0

    for prompt_data in data.get("per_prompt", []):
        for run in prompt_data.get("speculative", []):
            prof = run.get("profiling", {})
            if prof:
                phases["Draft Model"] += prof.get("draft_pct", 0)
                phases["Target Model"] += prof.get("target_pct", 0)
                phases["Sampling"] += prof.get("sampling_pct", 0)
                phases["Cache"] += prof.get("cache_pct", 0)
                phases["Overhead"] += prof.get("overhead_pct", 0)
                count += 1

    if count == 0:
        print("  Skipped: profiling_breakdown.png (no profiling data in benchmark)")
        return

    # Average
    for k in phases:
        phases[k] /= count

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9E9E9E"]
    labels = list(phases.keys())
    values = list(phases.values())

    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    # Add percentage labels on bars
    for bar, val in zip(bars, values):
        if val > 3:
            ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", ha="center", va="center", fontweight="bold",
                    color="white", fontsize=12)

    ax.set_xlabel("Percentage of Total Time")
    ax.set_title("Speculative Decoding: Where Time Goes")
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis="x")

    plt.savefig(os.path.join(output_dir, "profiling_breakdown.png"))
    plt.close()
    print("  Generated: profiling_breakdown.png")


def generate_all_plots(sweep_csv: str, output_dir: str = "plots", benchmark_json: str = None):
    """Generate all plots from a sweep results CSV and optional benchmark JSON."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading sweep data from {sweep_csv}...")
    df = pd.read_csv(sweep_csv)
    print(f"  {len(df)} data points loaded")

    print("\nGenerating plots...")
    plot_speedup_vs_k(df, output_dir)
    plot_acceptance_rate_vs_k(df, output_dir)
    plot_acceptance_heatmap(df, output_dir)
    plot_per_category_speedup(df, output_dir)
    plot_tokens_per_target_call(df, output_dir)
    plot_speedup_distribution(df, output_dir)

    if benchmark_json:
        plot_profiling_breakdown(benchmark_json, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from sweep results")
    parser.add_argument("sweep_csv", help="Path to sweep results CSV file")
    parser.add_argument("--benchmark-json", default=None, help="Benchmark JSON with profiling data")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    generate_all_plots(args.sweep_csv, args.output_dir, args.benchmark_json)


if __name__ == "__main__":
    main()
