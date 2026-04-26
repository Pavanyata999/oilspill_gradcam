#!/usr/bin/env python3
"""
Regenerate metrics-only IEEE assets (table + heatmap + bars) from an existing
metrics_table.csv inside results/ieee_results_package.

This does not touch ROC/confusion images (those require underlying predictions).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_metrics(csv_path: Path):
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    model_names = [r["Model"] for r in rows]
    metric_names = [k for k in rows[0].keys() if k != "Model"]
    values = np.array([[float(r[m]) for m in metric_names] for r in rows], dtype=np.float32)
    return model_names, metric_names, values


def save_table_png(out_path: Path, model_names, metric_names, values):
    headers = ["Model"] + metric_names
    cell_text = []
    for i, name in enumerate(model_names):
        row = [name] + [f"{values[i, j]:.4f}" for j in range(values.shape[1])]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(15, 2.8))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    plt.title("Performance Metrics Table", fontsize=16, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_png(out_path: Path, model_names, metric_names, values):
    fig, ax = plt.subplots(figsize=(12.8, 4.6))
    im = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(metric_names)), metric_names, rotation=20)
    ax.set_yticks(range(len(model_names)), model_names)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Score")
    ax.set_title("Metrics Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_bars_png(out_path: Path, model_names, metric_names, values):
    # Small 2x4 grid of the most common metrics for a paper figure.
    wanted = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "IoU", "Dice", "ROC-AUC"]
    picked = [m for m in wanted if m in metric_names]
    if not picked:
        picked = metric_names[:8]

    idx = [metric_names.index(m) for m in picked]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    x = np.arange(len(model_names))
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for ax, metric, j in zip(axes, picked, idx):
        vals = values[:, j]
        ax.bar(x, vals, color=colors[: len(model_names)], edgecolor="black")
        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(x, model_names, rotation=10)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = Path(__file__).parent
    pkg = base / "results" / "ieee_results_package"
    csv_path = pkg / "metrics_table.csv"

    model_names, metric_names, values = read_metrics(csv_path)
    save_table_png(pkg / "metrics_table.png", model_names, metric_names, values)
    save_heatmap_png(pkg / "metrics_heatmap.png", model_names, metric_names, values)
    save_bars_png(pkg / "metrics_comparison_chart.png", model_names, metric_names, values)


if __name__ == "__main__":
    main()

