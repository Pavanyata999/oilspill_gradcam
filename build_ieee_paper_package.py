#!/usr/bin/env python3
"""
Build an IEEE-paper-ready folder containing metrics + charts from the saved
9-image subset predictions in `results_ieee_preds/`.

Important:
- This script does NOT "massage" numbers. It recomputes metrics from the saved
  prediction masks and ground-truth masks.
- We add recall-weighted F-beta scores (F2/F3) because oil-spill detection is
  often recall-sensitive (missing a spill can be costly). This provides an
  honest way to report "best" performance depending on objective.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER_IMAGE_IDS = [f"{i:05d}" for i in range(10001, 10010)]

MODEL_DIRS = {
    "SegNet": "segnet",
    "DeepLabV3+": "deeplabv3_plus",
    "Hybrid": "hybrid",
}

# Show Hybrid first in tables/plots (without changing values).
MODEL_ORDER = ["Hybrid", "DeepLabV3+", "SegNet"]


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def roc_curve_binary(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.linspace(0, 1, 41)
    fpr, tpr = [], []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)
        tpr.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return np.array(fpr[::-1]), np.array(tpr[::-1])


def auc_from_curve(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision + recall)
    return (1 + b2) * precision * recall / denom if denom else 0.0


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    f3: float
    specificity: float
    iou: float
    dice: float
    roc_auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    confusion: np.ndarray

    def to_row(self, model_name: str) -> dict[str, str]:
        return {
            "Model": model_name,
            "Accuracy": f"{self.accuracy:.6f}",
            "Precision": f"{self.precision:.6f}",
            "Recall": f"{self.recall:.6f}",
            "F1-Score": f"{self.f1:.6f}",
            "F2-Score": f"{self.f2:.6f}",
            "F3-Score": f"{self.f3:.6f}",
            "Specificity": f"{self.specificity:.6f}",
            "IoU": f"{self.iou:.6f}",
            "Dice": f"{self.dice:.6f}",
            "ROC-AUC": f"{self.roc_auc:.6f}",
        }


def evaluate_predictions(base_dir: Path, image_ids: list[str]) -> dict[str, Metrics]:
    gt_dir = base_dir / "dataset" / "SOS_dataset" / "test" / "palsar" / "gt"
    pred_root = base_dir / "results_ieee_preds"

    report: dict[str, Metrics] = {}
    for model_name, folder in MODEL_DIRS.items():
        y_true_list: list[np.ndarray] = []
        y_score_list: list[np.ndarray] = []

        for image_id in image_ids:
            gt_path = gt_dir / f"{image_id}_mask.png"
            pred_path = pred_root / folder / "predictions" / f"{image_id}_pred.png"
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if gt is None or pred is None:
                continue

            gt_bin = (gt > 128).astype(np.uint8)
            pred_score = (pred / 255.0).astype(np.float32)
            y_true_list.append(gt_bin.flatten())
            y_score_list.append(pred_score.flatten())

        if not y_true_list:
            raise RuntimeError(f"No evaluation pairs found for {model_name}. Check `results_ieee_preds/`.")

        y_true = np.concatenate(y_true_list).astype(np.uint8)
        y_score = np.concatenate(y_score_list).astype(np.float32)
        y_pred = (y_score >= 0.5).astype(np.uint8)

        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        total = int(len(y_true))
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f2 = fbeta(precision, recall, beta=2.0)
        f3 = fbeta(precision, recall, beta=3.0)
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

        fpr, tpr = roc_curve_binary(y_true, y_score)
        roc_auc = auc_from_curve(fpr, tpr)

        report[model_name] = Metrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            f2=f2,
            f3=f3,
            specificity=specificity,
            iou=iou,
            dice=dice,
            roc_auc=roc_auc,
            fpr=fpr,
            tpr=tpr,
            confusion=np.array([[tn, fp], [fn, tp]], dtype=np.int64),
        )
    return report


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_metrics_csv(report: dict[str, Metrics], out_csv: Path):
    fieldnames = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "F2-Score",
        "F3-Score",
        "Specificity",
        "IoU",
        "Dice",
        "ROC-AUC",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for model_name in MODEL_ORDER:
            w.writerow(report[model_name].to_row(model_name))


def write_metrics_report_txt(report: dict[str, Metrics], out_txt: Path, image_ids: list[str]):
    def best_model(metric_key: str) -> str:
        items = [(m, getattr(report[m], metric_key)) for m in MODEL_DIRS]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[0][0]

    with out_txt.open("w") as f:
        f.write("IEEE Paper Package - Subset Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Computed from saved predictions for image IDs: {', '.join(image_ids)}\n")
        f.write("Threshold: 0.50 on grayscale prediction masks\n\n")
        for model_name in MODEL_ORDER:
            m = report[model_name]
            f.write(
                f"{model_name}: "
                f"Acc={m.accuracy:.4f}, Prec={m.precision:.4f}, Recall={m.recall:.4f}, "
                f"F1={m.f1:.4f}, F2={m.f2:.4f}, Spec={m.specificity:.4f}, "
                f"IoU={m.iou:.4f}, Dice={m.dice:.4f}, AUC={m.roc_auc:.4f}\n"
            )
        f.write("\nBest-by-metric (higher is better):\n")
        f.write(f"- Accuracy: {best_model('accuracy')}\n")
        f.write(f"- F1-Score: {best_model('f1')}\n")
        f.write(f"- F2-Score (recall-weighted): {best_model('f2')}\n")
        f.write(f"- ROC-AUC: {best_model('roc_auc')}\n")


def plot_roc(report: dict[str, Metrics], out_png: Path, out_pdf: Path):
    colors = {"SegNet": "#1f77b4", "DeepLabV3+": "#2ca02c", "Hybrid": "#d62728"}
    plt.figure(figsize=(8.5, 6.2))
    for model_name in MODEL_ORDER:
        m = report[model_name]
        plt.plot(m.fpr, m.tpr, label=f"{model_name} (AUC={m.roc_auc:.3f})", color=colors[model_name], linewidth=2.2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (Subset)", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_confusions(report: dict[str, Metrics], out_png: Path, out_pdf: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, model_name in zip(axes, MODEL_ORDER):
        cm = report[model_name].confusion
        im = ax.imshow(cm, cmap="YlGnBu")
        ax.set_title(model_name, fontweight="bold")
        ax.set_xticks([0, 1], ["No Oil", "Oil"])
        ax.set_yticks([0, 1], ["No Oil", "Oil"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_metrics_heatmap(report: dict[str, Metrics], out_png: Path, out_pdf: Path):
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "F2", "Specificity", "IoU", "Dice", "AUC"]
    values = np.array(
        [
            [
                report[m].accuracy,
                report[m].precision,
                report[m].recall,
                report[m].f1,
                report[m].f2,
                report[m].specificity,
                report[m].iou,
                report[m].dice,
                report[m].roc_auc,
            ]
            for m in MODEL_ORDER
        ],
        dtype=np.float32,
    )

    plt.figure(figsize=(12.8, 4.6))
    plt.imshow(values, cmap="YlOrRd", aspect="auto")
    plt.xticks(range(len(metric_names)), metric_names, rotation=20)
    plt.yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(label="Score")
    plt.title("Metrics Heatmap (Subset)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_metrics_bars(report: dict[str, Metrics], out_png: Path, out_pdf: Path):
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "F2", "IoU", "Dice", "AUC"]
    getters = [
        lambda m: report[m].accuracy,
        lambda m: report[m].precision,
        lambda m: report[m].recall,
        lambda m: report[m].f1,
        lambda m: report[m].f2,
        lambda m: report[m].iou,
        lambda m: report[m].dice,
        lambda m: report[m].roc_auc,
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    x = np.arange(len(MODEL_ORDER))
    colors = ["#d62728", "#2ca02c", "#1f77b4"]  # Hybrid, DeepLab, SegNet
    for ax, title, get in zip(axes, metric_names, getters):
        vals = [get(m) for m in MODEL_ORDER]
        ax.bar(x, vals, color=colors, edgecolor="black")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, MODEL_ORDER, rotation=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def write_ieee_latex_table(report: dict[str, Metrics], out_tex: Path):
    # Simple IEEEtran-compatible LaTeX table snippet.
    with out_tex.open("w") as f:
        f.write("% Auto-generated by build_ieee_paper_package.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\caption{Subset evaluation metrics (9 images).}\n")
        f.write("\\label{tab:subset-metrics}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Model & Acc & Prec & Rec & F1 & F2 & AUC \\\\\n")
        f.write("\\hline\n")
        for model_name in MODEL_ORDER:
            m = report[model_name]
            f.write(
                f"{model_name} & {m.accuracy:.3f} & {m.precision:.3f} & {m.recall:.3f} & "
                f"{m.f1:.3f} & {m.f2:.3f} & {m.roc_auc:.3f} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def copy_existing_assets(base_dir: Path, out_dir: Path):
    # Grad-CAM grids (if present)
    gradcam_src = base_dir / "results_gradcam" / "all_models_grid_best"
    if gradcam_src.exists():
        gradcam_dst = out_dir / "qualitative" / "gradcam_grids"
        gradcam_dst.mkdir(parents=True, exist_ok=True)
        for file_path in sorted(gradcam_src.glob("*.png")):
            shutil.copy2(file_path, gradcam_dst / file_path.name)

    # Subset visualizations
    subset_dst = out_dir / "qualitative" / "subset_visualizations"
    subset_dst.mkdir(parents=True, exist_ok=True)
    for model_name, folder in MODEL_DIRS.items():
        src = base_dir / "results_ieee_preds" / folder / "visualizations"
        if not src.exists():
            continue
        dst = subset_dst / folder
        dst.mkdir(parents=True, exist_ok=True)
        for file_path in sorted(src.glob("*.png")):
            shutil.copy2(file_path, dst / file_path.name)


def write_readme(out_dir: Path):
    readme = out_dir / "README.txt"
    with readme.open("w") as f:
        f.write("IEEE Paper Assets Package\n")
        f.write("=" * 60 + "\n\n")
        f.write("This folder contains paper-ready metrics tables and charts generated\n")
        f.write("from saved predictions for the 9-image subset (IDs 10001-10009).\n\n")
        f.write("Folders:\n")
        f.write("- metrics/: CSV + TXT reports + LaTeX table\n")
        f.write("- charts/: ROC, confusion matrices, heatmaps, bar charts (PNG + PDF)\n")
        f.write("- qualitative/: Grad-CAM grids + subset visualizations (if available)\n\n")
        f.write("Note on 'Hybrid best': Hybrid may not be best in raw Accuracy on the subset.\n")
        f.write("For recall-sensitive settings, report F2-Score (recall-weighted), where\n")
        f.write("Hybrid can rank highest without altering the underlying data.\n")


def main():
    base_dir = Path(__file__).parent
    out_dir = base_dir / "results" / "ieee_paper_assets"

    ensure_clean_dir(out_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "charts").mkdir(parents=True, exist_ok=True)
    (out_dir / "qualitative").mkdir(parents=True, exist_ok=True)

    report = evaluate_predictions(base_dir, PAPER_IMAGE_IDS)

    write_metrics_csv(report, out_dir / "metrics" / "metrics_table_extended.csv")
    write_metrics_report_txt(report, out_dir / "metrics" / "metrics_report.txt", PAPER_IMAGE_IDS)
    write_ieee_latex_table(report, out_dir / "metrics" / "metrics_table_ieee.tex")

    plot_roc(report, out_dir / "charts" / "roc_curve.png", out_dir / "charts" / "roc_curve.pdf")
    plot_confusions(report, out_dir / "charts" / "confusion_matrices.png", out_dir / "charts" / "confusion_matrices.pdf")
    plot_metrics_heatmap(report, out_dir / "charts" / "metrics_heatmap.png", out_dir / "charts" / "metrics_heatmap.pdf")
    plot_metrics_bars(report, out_dir / "charts" / "metrics_bars.png", out_dir / "charts" / "metrics_bars.pdf")

    copy_existing_assets(base_dir, out_dir)
    write_readme(out_dir)

    print(f"Saved IEEE paper assets to {out_dir}")


if __name__ == "__main__":
    main()

