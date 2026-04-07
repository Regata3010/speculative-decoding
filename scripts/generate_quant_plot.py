#!/usr/bin/env python3
"""Generate plot from quantization experiment results."""

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

OUTPUT_DIR = "plots/quantization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardcoded results from experiment
data = [
    {"draft": "FP16", "target": "INT8", "speedup": 2.20, "accept": 0.778, "vram": 90.5, "base": 5.2},
    {"draft": "INT8", "target": "INT8", "speedup": 1.42, "accept": 0.808, "vram": 84.0, "base": 5.6},
    {"draft": "INT4", "target": "INT8", "speedup": 2.56, "accept": 0.757, "vram": 81.1, "base": 5.6},
    {"draft": "FP16", "target": "INT4", "speedup": 1.00, "accept": 0.749, "vram": 59.8, "base": 21.3},
    {"draft": "INT8", "target": "INT4", "speedup": 0.44, "accept": 0.741, "vram": 53.3, "base": 21.3},
    {"draft": "INT4", "target": "INT4", "speedup": 1.00, "accept": 0.772, "vram": 50.4, "base": 21.3},
]


def plot_speedup_matrix():
    """Heatmap: draft precision vs target precision → speedup."""
    fig, ax = plt.subplots(figsize=(8, 5))

    drafts = ["FP16", "INT8", "INT4"]
    targets = ["INT8", "INT4"]

    matrix = np.zeros((len(drafts), len(targets)))
    for d in data:
        i = drafts.index(d["draft"])
        j = targets.index(d["target"])
        matrix[i, j] = d["speedup"]

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=2.8)

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_yticks(range(len(drafts)))
    ax.set_yticklabels(drafts)
    ax.set_xlabel("Target Precision")
    ax.set_ylabel("Draft Precision")
    ax.set_title("Speedup by Quantization Configuration — Qwen 72B + 7B (H200)")

    for i in range(len(drafts)):
        for j in range(len(targets)):
            val = matrix[i, j]
            color = "white" if val > 1.8 or val < 0.6 else "black"
            ax.text(j, i, f"{val:.2f}x", ha="center", va="center",
                    fontweight="bold", fontsize=14, color=color)

    plt.colorbar(im, ax=ax, label="Speedup")
    plt.savefig(os.path.join(OUTPUT_DIR, "speedup_matrix.png"))
    plt.close()
    print("  Generated: speedup_matrix.png")


def plot_speedup_vs_vram():
    """Scatter: VRAM usage vs speedup, showing the efficiency frontier."""
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {"INT8": "#2196F3", "INT4": "#FF5722"}

    for d in data:
        color = colors[d["target"]]
        marker = "o" if d["target"] == "INT8" else "s"
        ax.scatter(d["vram"], d["speedup"], s=200, c=color, marker=marker,
                   edgecolors="black", linewidth=1, zorder=5)
        label = f'{d["draft"]}\n+{d["target"]}'
        offset_x = 1.5
        offset_y = 0.05 if d["speedup"] > 1 else -0.1
        ax.annotate(label, (d["vram"], d["speedup"]),
                    textcoords="offset points", xytext=(offset_x * 10, offset_y * 100),
                    fontsize=8, ha="center")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(52, 1.03, "break-even", fontsize=9, color="gray")

    # Highlight best config
    best = [d for d in data if d["speedup"] == 2.56][0]
    ax.annotate("BEST", (best["vram"], best["speedup"]),
                textcoords="offset points", xytext=(-40, 20),
                fontsize=11, fontweight="bold", color="#4CAF50",
                arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=2))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="INT8 target"),
        Patch(facecolor="#FF5722", label="INT4 target"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_xlabel("GPU VRAM Usage (GB)")
    ax.set_ylabel("Speedup over Baseline")
    ax.set_title("Speedup vs VRAM: Quantization Efficiency Frontier")
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, "speedup_vs_vram.png"))
    plt.close()
    print("  Generated: speedup_vs_vram.png")


def plot_speed_vs_accuracy():
    """Bar chart: speedup and acceptance rate side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = [f"{d['draft']}→{d['target']}" for d in data]
    speedups = [d["speedup"] for d in data]
    accepts = [d["accept"] * 100 for d in data]

    colors = ["#4CAF50" if s > 1.0 else "#FF5722" for s in speedups]

    # Speedup
    bars1 = ax1.bar(labels, speedups, color=colors, edgecolor="white")
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5)
    for bar, val in zip(bars1, speedups):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f"{val:.2f}x", ha="center", fontweight="bold", fontsize=9)
    ax1.set_ylabel("Speedup")
    ax1.set_title("Speedup by Config")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(True, alpha=0.3, axis="y")

    # Acceptance rate
    bars2 = ax2.bar(labels, accepts, color="#2196F3", edgecolor="white")
    for bar, val in zip(bars2, accepts):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f"{val:.1f}%", ha="center", fontweight="bold", fontsize=9)
    ax2.set_ylabel("Acceptance Rate (%)")
    ax2.set_title("Acceptance Rate by Config")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Quantization Experiment: Qwen 72B + 7B on H200", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "speed_vs_accuracy.png"))
    plt.close()
    print("  Generated: speed_vs_accuracy.png")


if __name__ == "__main__":
    print("Generating quantization experiment plots...")
    plot_speedup_matrix()
    plot_speedup_vs_vram()
    plot_speed_vs_accuracy()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")
