#!/usr/bin/env python3
"""
Build a paper-ready results package from saved 9-image predictions and Grad-CAM grids.
"""

import csv
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_DIRS = {
    "SegNet": "segnet",
    "DeepLabV3+": "deeplabv3_plus",
    "Hybrid": "hybrid",
}


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def roc_curve_binary(y_true: np.ndarray, y_score: np.ndarray):
    thresholds = np.linspace(0, 1, 21)
    fpr, tpr = [], []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)
        tpr.append(tp / (tp + fn) if (tp + fn) else 0.0)
    fpr = np.array(fpr[::-1])
    tpr = np.array(tpr[::-1])
    return fpr, tpr


def auc_from_curve(fpr: np.ndarray, tpr: np.ndarray):
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def evaluate_predictions(base_dir: Path):
    gt_dir = base_dir / "dataset" / "SOS_dataset" / "test" / "palsar" / "gt"
    pred_root = base_dir / "results_ieee_preds"
    image_ids = [f"{i:05d}" for i in range(10001, 10010)]

    report = {}
    for model_name, folder in MODEL_DIRS.items():
        y_true_list = []
        y_score_list = []
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

        y_true = np.concatenate(y_true_list)
        y_score = np.concatenate(y_score_list)
        y_pred = (y_score >= 0.5).astype(np.uint8)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        fpr, tpr = roc_curve_binary(y_true, y_score)
        auc = auc_from_curve(fpr, tpr)
        report[model_name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Specificity": spec,
            "IoU": iou,
            "Dice": dice,
            "ROC-AUC": auc,
            "FPR": fpr,
            "TPR": tpr,
            "Confusion": np.array([[tn, fp], [fn, tp]]),
        }
    return report


def save_metrics_table(report: dict, output_dir: Path):
    csv_path = output_dir / "metrics_table.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "IoU", "Dice", "ROC-AUC"])
        for model_name, metrics in report.items():
            writer.writerow([
                model_name,
                f"{metrics['Accuracy']:.6f}",
                f"{metrics['Precision']:.6f}",
                f"{metrics['Recall']:.6f}",
                f"{metrics['F1-Score']:.6f}",
                f"{metrics['Specificity']:.6f}",
                f"{metrics['IoU']:.6f}",
                f"{metrics['Dice']:.6f}",
                f"{metrics['ROC-AUC']:.6f}",
            ])

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    rows = []
    for model_name, metrics in report.items():
        rows.append([
            model_name,
            f"{metrics['Accuracy']:.4f}",
            f"{metrics['Precision']:.4f}",
            f"{metrics['Recall']:.4f}",
            f"{metrics['F1-Score']:.4f}",
            f"{metrics['Specificity']:.4f}",
            f"{metrics['IoU']:.4f}",
            f"{metrics['Dice']:.4f}",
            f"{metrics['ROC-AUC']:.4f}",
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "Acc", "Prec", "Recall", "F1", "Spec", "IoU", "Dice", "AUC"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)
    plt.title("Metrics Table on 9-Image Paper Subset", fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    txt_path = output_dir / "metrics_report.txt"
    with txt_path.open("w") as handle:
        handle.write("IEEE Results Package - 9 Image Paper Subset\n")
        handle.write("=" * 60 + "\n")
        handle.write("Quantitative metrics are computed from the saved prediction masks\n")
        handle.write("for image IDs 10001 to 10009.\n\n")
        for model_name, metrics in report.items():
            handle.write(
                f"{model_name}: Acc={metrics['Accuracy']:.4f}, Prec={metrics['Precision']:.4f}, "
                f"Recall={metrics['Recall']:.4f}, F1={metrics['F1-Score']:.4f}, "
                f"Spec={metrics['Specificity']:.4f}, IoU={metrics['IoU']:.4f}, "
                f"Dice={metrics['Dice']:.4f}, AUC={metrics['ROC-AUC']:.4f}\n"
            )


def save_roc(report: dict, output_dir: Path):
    colors = {"SegNet": "#1f77b4", "DeepLabV3+": "#2ca02c", "Hybrid": "#d62728"}
    plt.figure(figsize=(8.5, 6.2))
    for model_name, metrics in report.items():
        plt.plot(metrics["FPR"], metrics["TPR"], label=f"{model_name} (AUC={metrics['ROC-AUC']:.3f})", color=colors[model_name], linewidth=2.2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_confusions(report: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (model_name, metrics) in zip(axes, report.items()):
        cm = metrics["Confusion"]
        im = ax.imshow(cm, cmap="YlGnBu")
        ax.set_title(model_name, fontweight="bold")
        ax.set_xticks([0, 1], ["No Oil", "Oil"])
        ax.set_yticks([0, 1], ["No Oil", "Oil"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_metric_charts(report: dict, output_dir: Path):
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "IoU", "Dice", "ROC-AUC"]
    values = np.array([[report[model][metric] for metric in metric_names] for model in MODEL_DIRS])

    plt.figure(figsize=(12, 4.8))
    plt.imshow(values, cmap="YlOrRd", aspect="auto")
    plt.xticks(range(len(metric_names)), metric_names, rotation=20)
    plt.yticks(range(len(MODEL_DIRS)), list(MODEL_DIRS.keys()))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(label="Score")
    plt.title("Metrics Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    x = np.arange(len(MODEL_DIRS))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for ax, metric in zip(axes, metric_names):
        vals = [report[model][metric] for model in MODEL_DIRS]
        ax.bar(x, vals, color=colors, edgecolor="black")
        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(x, list(MODEL_DIRS.keys()), rotation=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison_chart.png", dpi=300, bbox_inches="tight")
    plt.close()


def copy_assets(base_dir: Path, output_dir: Path):
    gradcam_src = base_dir / "results_gradcam" / "all_models_grid_best"
    gradcam_dst = output_dir / "gradcam_grids"
    gradcam_dst.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(gradcam_src.glob("*.png")):
        shutil.copy2(file_path, gradcam_dst / file_path.name)

    qual_dst = output_dir / "subset_visualizations"
    qual_dst.mkdir(parents=True, exist_ok=True)
    for folder in MODEL_DIRS.values():
        model_dst = qual_dst / folder
        model_dst.mkdir(parents=True, exist_ok=True)
        src_pred = base_dir / "results_ieee_preds" / folder / "visualizations"
        for file_path in sorted(src_pred.glob("*.png")):
            shutil.copy2(file_path, model_dst / file_path.name)


def save_readme(output_dir: Path):
    readme = output_dir / "README_RESULTS_PACKAGE.txt"
    with readme.open("w") as handle:
        handle.write("IEEE Research Paper Results Package\n")
        handle.write("=" * 50 + "\n")
        handle.write("This folder contains the final subset-based paper outputs and the\n")
        handle.write("Grad-CAM grid extension figures.\n\n")
        handle.write("Main files:\n")
        handle.write("- roc_curves_comparison.png\n")
        handle.write("- confusion_matrices.png\n")
        handle.write("- metrics_table.csv\n")
        handle.write("- metrics_table.png\n")
        handle.write("- metrics_heatmap.png\n")
        handle.write("- metrics_comparison_chart.png\n")
        handle.write("- metrics_report.txt\n")
        handle.write("- gradcam_grids/\n")
        handle.write("- subset_visualizations/\n")


def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "results" / "ieee_results_package"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate_predictions(base_dir)
    save_metrics_table(report, output_dir)
    save_roc(report, output_dir)
    save_confusions(report, output_dir)
    save_metric_charts(report, output_dir)
    copy_assets(base_dir, output_dir)
    save_readme(output_dir)

    print(f"IEEE subset package saved to {output_dir}")


if __name__ == "__main__":
    main()
