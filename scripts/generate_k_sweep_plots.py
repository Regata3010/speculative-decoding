#!/usr/bin/env python3
"""Generate plots comparing INT4 vs FP16 draft K-sweeps with per-position analysis."""

import os
import sys

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

OUTPUT_DIR = "plots/k_sweep"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# INT4 draft results
int4 = {
    "K":       [1,    2,    3,    5,    7,    10,   15],
    "speedup": [1.30, 1.62, 1.94, 2.23, 2.48, 2.51, 2.81],
    "accept":  [88.4, 85.1, 81.1, 73.6, 68.6, 62.0, 55.5],
    "tps":     [7.1,  9.3,  11.1, 12.8, 14.2, 14.0, 16.4],
    "per_pos": {
        1:  [92],
        2:  [91, 84],
        3:  [90, 84, 78],
        5:  [89, 80, 73, 69, 64],
        7:  [87, 78, 73, 65, 61, 58, 54],
        10: [87, 78, 71, 65, 60, 56, 50, 48, 44, 40],
        15: [84, 74, 70, 61, 55, 51, 47, 44, 41, 40, 38, 37, 37, 35, 34],
    }
}

# FP16 draft results
fp16 = {
    "K":       [1,    2,    3,    5,    7,    10,   15],
    "speedup": [1.34, 1.79, 2.02, 2.50, 2.73, 2.93, 3.30],
    "accept":  [88.7, 84.9, 80.6, 75.2, 69.4, 61.6, 55.6],
    "tps":     [7.1,  9.6,  10.9, 13.6, 14.9, 15.7, 17.6],
    "per_pos": {
        1:  [92],
        2:  [90, 83],
        3:  [90, 83, 79],
        5:  [90, 82, 76, 70, 67],
        7:  [86, 76, 69, 64, 60, 57, 54],
        10: [90, 81, 76, 72, 66, 64, 59, 56, 54, 52],
        15: [84, 71, 65, 59, 54, 51, 45, 43, 40, 38, 37, 36, 35, 35, 33],
    }
}

BASELINE_TPS = 5.5  # Average across runs


def plot_speedup_vs_k():
    """Hero plot: speedup vs K for both draft precisions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(fp16["K"], fp16["speedup"], "o-", linewidth=2.5, markersize=8,
            color="#2196F3", label="FP16 draft + INT8 target")
    ax.plot(int4["K"], int4["speedup"], "s--", linewidth=2.5, markersize=8,
            color="#FF9800", label="INT4 draft + INT8 target")
    ax.axhline(y=1.0, color="#FF5722", linestyle=":", linewidth=1.5,
               label="1x (baseline)", alpha=0.7)

    # Annotate best
    ax.annotate(f'3.30x', (15, 3.30), textcoords="offset points",
                xytext=(-40, 10), fontsize=12, fontweight="bold", color="#2196F3")
    ax.annotate(f'2.81x', (15, 2.81), textcoords="offset points",
                xytext=(-40, -20), fontsize=12, fontweight="bold", color="#FF9800")

    ax.set_xlabel("K (draft tokens per step)")
    ax.set_ylabel("Speedup over Baseline")
    ax.set_title("Speculative Decoding Speedup vs K — Qwen 72B (INT8) on H200")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fp16["K"])

    plt.savefig(os.path.join(OUTPUT_DIR, "speedup_vs_k.png"))
    plt.close()
    print("  Generated: speedup_vs_k.png")


def plot_acceptance_vs_k():
    """Acceptance rate vs K."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(fp16["K"], fp16["accept"], "o-", linewidth=2.5, markersize=8,
            color="#2196F3", label="FP16 draft")
    ax.plot(int4["K"], int4["accept"], "s--", linewidth=2.5, markersize=8,
            color="#FF9800", label="INT4 draft")

    ax.set_xlabel("K (draft tokens per step)")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_title("Draft Token Acceptance Rate vs K")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fp16["K"])
    ax.set_ylim(40, 100)

    plt.savefig(os.path.join(OUTPUT_DIR, "acceptance_vs_k.png"))
    plt.close()
    print("  Generated: acceptance_vs_k.png")


def plot_per_position_heatmap():
    """Heatmap of per-position acceptance rate for FP16 draft."""
    max_k = 15
    k_values = [1, 2, 3, 5, 7, 10, 15]

    matrix = np.full((len(k_values), max_k), np.nan)
    for i, k in enumerate(k_values):
        positions = fp16["per_pos"][k]
        for j, val in enumerate(positions):
            matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=30, vmax=95)

    ax.set_xticks(range(max_k))
    ax.set_xticklabels([f"pos{i}" for i in range(max_k)], fontsize=9)
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([f"K={k}" for k in k_values])
    ax.set_xlabel("Position in Draft Sequence")
    ax.set_ylabel("K value")
    ax.set_title("Per-Position Acceptance Rate (%) — FP16 Draft + INT8 Target")

    for i in range(len(k_values)):
        for j in range(max_k):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 50 or val > 85 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Acceptance Rate (%)", shrink=0.8)
    plt.savefig(os.path.join(OUTPUT_DIR, "per_position_heatmap.png"))
    plt.close()
    print("  Generated: per_position_heatmap.png")


def plot_per_position_decay():
    """Line plot showing acceptance decay across positions for select K values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {5: "#2196F3", 7: "#4CAF50", 10: "#FF9800", 15: "#E91E63"}

    for k in [5, 7, 10, 15]:
        positions = list(range(k))
        rates = fp16["per_pos"][k]
        ax.plot(positions, rates, "o-", linewidth=2, markersize=6,
                color=colors[k], label=f"K={k}")

    ax.axhline(y=50, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(14.5, 51, "50% (coin flip)", fontsize=9, color="gray", ha="right")

    ax.set_xlabel("Position in Draft Sequence (0 = first draft token)")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_title("How Acceptance Decays with Position — FP16 Draft + INT8 Target")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(25, 100)

    plt.savefig(os.path.join(OUTPUT_DIR, "per_position_decay.png"))
    plt.close()
    print("  Generated: per_position_decay.png")


def plot_fp16_vs_int4_comparison():
    """Side by side: FP16 vs INT4 speedup delta."""
    fig, ax = plt.subplots(figsize=(10, 6))

    K = fp16["K"]
    delta = [f - i for f, i in zip(fp16["speedup"], int4["speedup"])]

    bars = ax.bar([str(k) for k in K], delta, color="#4CAF50", edgecolor="white")
    for bar, val in zip(bars, delta):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"+{val:.2f}x", ha="center", fontweight="bold", fontsize=11)

    ax.set_xlabel("K (draft tokens per step)")
    ax.set_ylabel("FP16 Speedup Advantage over INT4")
    ax.set_title("FP16 vs INT4 Draft: Speedup Delta at Each K")
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(os.path.join(OUTPUT_DIR, "fp16_vs_int4_delta.png"))
    plt.close()
    print("  Generated: fp16_vs_int4_delta.png")


def plot_throughput_vs_k():
    """Raw throughput (tok/s) vs K."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(fp16["K"], fp16["tps"], "o-", linewidth=2.5, markersize=8,
            color="#2196F3", label="FP16 draft (speculative)")
    ax.plot(int4["K"], int4["tps"], "s--", linewidth=2.5, markersize=8,
            color="#FF9800", label="INT4 draft (speculative)")
    ax.axhline(y=BASELINE_TPS, color="#FF5722", linestyle="--", linewidth=2,
               label=f"Baseline ({BASELINE_TPS} tok/s)")

    ax.set_xlabel("K (draft tokens per step)")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput vs K — Qwen 72B (INT8 target) on H200")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fp16["K"])

    plt.savefig(os.path.join(OUTPUT_DIR, "throughput_vs_k.png"))
    plt.close()
    print("  Generated: throughput_vs_k.png")


if __name__ == "__main__":
    print("Generating K-sweep comparison plots...")
    plot_speedup_vs_k()
    plot_acceptance_vs_k()
    plot_per_position_heatmap()
    plot_per_position_decay()
    plot_fp16_vs_int4_comparison()
    plot_throughput_vs_k()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")
