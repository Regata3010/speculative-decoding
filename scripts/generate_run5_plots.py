#!/usr/bin/env python3
"""Generate plots from Run 5 benchmark results."""

import json
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = "plots/run5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all run5 benchmarks
LLAMA_70B = "run5 benchmark/benchmark_20260331_190109.json"
QWEN_72B_4BIT = "run5 benchmark/benchmark_20260331_200711.json"
QWEN_72B_8BIT = "run5 benchmark/benchmark_20260331_213122.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_per_prompt_speedups(data):
    results = []
    for p in data["per_prompt"]:
        stps = statistics.mean([r["tokens_per_second"] for r in p["speculative"]])
        btps = statistics.mean([r["tokens_per_second"] for r in p["baseline"]])
        if btps > 0:
            results.append({
                "name": p["prompt_name"],
                "category": p["category"],
                "spec_tps": stps,
                "base_tps": btps,
                "speedup": stps / btps,
            })
    return results


def get_avg_profiling(data):
    phases = {"Draft Model": [], "Target Model": [], "Sampling": [], "Overhead": []}
    for p in data["per_prompt"]:
        for r in p.get("speculative", []):
            prof = r.get("profiling", {})
            if prof:
                phases["Draft Model"].append(prof.get("draft_pct", 0))
                phases["Target Model"].append(prof.get("target_pct", 0))
                phases["Sampling"].append(prof.get("sampling_pct", 0))
                phases["Overhead"].append(prof.get("overhead_pct", 0))
    return {k: statistics.mean(v) if v else 0 for k, v in phases.items()}


# ============================================================
# Plot 1: Headline comparison — all 3 H200 configs
# ============================================================
def plot_headline_comparison():
    configs = []
    for path, label in [
        (LLAMA_70B, "Llama 70B\n(8-bit)"),
        (QWEN_72B_8BIT, "Qwen 72B\n(8-bit)"),
        (QWEN_72B_4BIT, "Qwen 72B\n(4-bit)"),
    ]:
        d = load_json(path)
        configs.append({
            "label": label,
            "spec": d["speculative_aggregate"]["tokens_per_sec_mean"],
            "base": d["baseline_aggregate"]["tokens_per_sec_mean"],
            "speedup": d["speculative_aggregate"]["speedup_mean"],
        })

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(configs))
    w = 0.35

    bars_spec = ax.bar(x - w/2, [c["spec"] for c in configs], w,
                       label="Speculative Decoding", color="#2196F3", edgecolor="white")
    bars_base = ax.bar(x + w/2, [c["base"] for c in configs], w,
                       label="Baseline", color="#FF5722", edgecolor="white")

    # Add speedup labels
    for i, c in enumerate(configs):
        ax.text(i - w/2, c["spec"] + 0.5, f'{c["speedup"]:.2f}x',
                ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Tokens / second")
    ax.set_title("Speculative Decoding vs Baseline on H200 (Run 5)")
    ax.set_xticks(x)
    ax.set_xticklabels([c["label"] for c in configs])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(os.path.join(OUTPUT_DIR, "headline_comparison.png"))
    plt.close()
    print("  Generated: headline_comparison.png")


# ============================================================
# Plot 2: Per-category speedup (Qwen 72B 8-bit — best config)
# ============================================================
def plot_per_category():
    data = load_json(QWEN_72B_8BIT)
    results = get_per_prompt_speedups(data)

    # Group by category
    from collections import defaultdict
    cats = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r["speedup"])

    cat_means = {k: statistics.mean(v) for k, v in cats.items()}
    cat_stds = {k: statistics.stdev(v) if len(v) > 1 else 0 for k, v in cats.items()}

    sorted_cats = sorted(cat_means.items(), key=lambda x: x[1])
    labels = [c[0] for c in sorted_cats]
    means = [c[1] for c in sorted_cats]
    stds = [cat_stds[c[0]] for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, means, xerr=stds, capsize=4,
                   color="#2196F3", alpha=0.85, edgecolor="white")
    ax.axvline(x=1.0, color="#FF5722", linestyle="--", linewidth=2, label="1x (no speedup)")

    for bar, val in zip(bars, means):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}x", va="center", fontweight="bold", fontsize=10)

    ax.set_xlabel("Speedup over Baseline")
    ax.set_title("Speculative Decoding Speedup by Category — Qwen 72B (8-bit) on H200")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.savefig(os.path.join(OUTPUT_DIR, "per_category_qwen72b.png"))
    plt.close()
    print("  Generated: per_category_qwen72b.png")


# ============================================================
# Plot 3: Per-prompt speedup (Qwen 72B 8-bit — all 22 prompts)
# ============================================================
def plot_per_prompt():
    data = load_json(QWEN_72B_8BIT)
    results = get_per_prompt_speedups(data)
    results.sort(key=lambda x: x["speedup"])

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "code_generation": "#2196F3",
        "summarization": "#4CAF50",
        "qa": "#FF9800",
        "creative_writing": "#E91E63",
        "reasoning": "#9C27B0",
        "translation": "#00BCD4",
    }

    bars = ax.barh(
        [r["name"] for r in results],
        [r["speedup"] for r in results],
        color=[colors.get(r["category"], "#999") for r in results],
        edgecolor="white",
    )
    ax.axvline(x=1.0, color="#FF5722", linestyle="--", linewidth=2, label="1x")

    ax.set_xlabel("Speedup over Baseline")
    ax.set_title("Speculative Decoding Speedup: All 22 Prompts — Qwen 72B (8-bit) on H200")
    ax.grid(True, alpha=0.3, axis="x")

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.savefig(os.path.join(OUTPUT_DIR, "per_prompt_qwen72b.png"))
    plt.close()
    print("  Generated: per_prompt_qwen72b.png")


# ============================================================
# Plot 4: Profiling comparison across runs
# ============================================================
def plot_profiling_evolution():
    """Show how the draft/target time split evolved across runs."""
    # Hardcoded from actual run data
    runs = [
        {"label": "Run 2\nQwen 7B\n(A100)", "draft": 60, "target": 29, "other": 11},
        {"label": "Run 2\nLlama 8B\n(A100)", "draft": 55.6, "target": 31.6, "other": 12.8},
        {"label": "Run 3\nQwen 72B 4-bit\n(A100)", "draft": 25.5, "target": 72.6, "other": 1.9},
        {"label": "Run 5\nLlama 70B 8-bit\n(H200)", "draft": 26.5, "target": 73.2, "other": 0.3},
        {"label": "Run 5\nQwen 72B 8-bit\n(H200)", "draft": 22.3, "target": 77.4, "other": 0.3},
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(runs))

    draft = [r["draft"] for r in runs]
    target = [r["target"] for r in runs]
    other = [r["other"] for r in runs]

    ax.bar(x, draft, label="Draft Model", color="#2196F3")
    ax.bar(x, target, bottom=draft, label="Target Model", color="#FF5722")
    ax.bar(x, other, bottom=[d+t for d, t in zip(draft, target)],
           label="Sampling + Overhead", color="#9E9E9E")

    ax.set_ylabel("Percentage of Total Time")
    ax.set_title("Where Time Goes: Profiling Evolution Across Runs")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in runs], fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    # Add speedup annotation
    speedups = ["0.75x", "0.99x", "1.08x", "2.16x", "2.60x"]
    for i, sp in enumerate(speedups):
        ax.text(i, 102, sp, ha="center", fontweight="bold", fontsize=11,
                color="#2196F3" if float(sp[:-1]) > 1 else "#FF5722")

    plt.savefig(os.path.join(OUTPUT_DIR, "profiling_evolution.png"))
    plt.close()
    print("  Generated: profiling_evolution.png")


# ============================================================
# Plot 5: Profiling breakdown for best config
# ============================================================
def plot_profiling_best():
    data = load_json(QWEN_72B_8BIT)
    phases = get_avg_profiling(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9E9E9E"]
    labels = list(phases.keys())
    values = list(phases.values())

    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        if val > 2:
            ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", ha="center", va="center",
                    fontweight="bold", color="white", fontsize=13)

    ax.set_xlabel("Percentage of Total Time")
    ax.set_title("Speculative Decoding: Where Time Goes — Qwen 72B (8-bit) on H200")
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis="x")

    plt.savefig(os.path.join(OUTPUT_DIR, "profiling_qwen72b_8bit.png"))
    plt.close()
    print("  Generated: profiling_qwen72b_8bit.png")


# ============================================================
# Plot 6: 4-bit vs 8-bit quantization comparison
# ============================================================
def plot_quantization_comparison():
    d4 = load_json(QWEN_72B_4BIT)
    d8 = load_json(QWEN_72B_8BIT)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: throughput
    labels = ["4-bit", "8-bit"]
    spec = [d4["speculative_aggregate"]["tokens_per_sec_mean"],
            d8["speculative_aggregate"]["tokens_per_sec_mean"]]
    base = [d4["baseline_aggregate"]["tokens_per_sec_mean"],
            d8["baseline_aggregate"]["tokens_per_sec_mean"]]

    x = np.arange(2)
    w = 0.35
    ax1.bar(x - w/2, spec, w, label="Speculative", color="#2196F3")
    ax1.bar(x + w/2, base, w, label="Baseline", color="#FF5722")
    ax1.set_ylabel("Tokens / second")
    ax1.set_title("Throughput: 4-bit vs 8-bit")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: speedup
    speedups = [d4["speculative_aggregate"]["speedup_mean"],
                d8["speculative_aggregate"]["speedup_mean"]]
    bars = ax2.bar(labels, speedups, color=["#FF9800", "#4CAF50"], edgecolor="white")
    ax2.axhline(y=1.0, color="#FF5722", linestyle="--", linewidth=2)
    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f"{val:.2f}x", ha="center", fontweight="bold", fontsize=13)
    ax2.set_ylabel("Speedup")
    ax2.set_title("Speedup: 4-bit vs 8-bit")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Qwen 72B: Why Quantization Level Matters", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "quantization_comparison.png"))
    plt.close()
    print("  Generated: quantization_comparison.png")


if __name__ == "__main__":
    print("Generating Run 5 plots...")
    plot_headline_comparison()
    plot_per_category()
    plot_per_prompt()
    plot_profiling_evolution()
    plot_profiling_best()
    plot_quantization_comparison()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")
