#!/usr/bin/env python3
"""
Lightweight IEEE results package generator using pure NumPy + Matplotlib.
Avoids heavier dependencies while still using real model outputs.
"""

import csv
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_evaluation_metrics import (
    HybridOilSpillModel,
    get_device,
    get_predictions_with_segmentation,
    get_predictions_without_segmentation,
    load_ground_truth,
    preprocess_image,
)


MODEL_NAMES = ["SegNet", "DeepLabV3+", "Hybrid"]


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tn, fp, fn, tp = compute_confusion(y_true, y_pred)

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    fpr, tpr = roc_curve_from_probs(y_true, y_prob)
    roc_auc = auc_from_curve(fpr, tpr)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Specificity": specificity,
        "IoU": iou,
        "Dice": dice,
        "ROC-AUC": roc_auc,
        "Confusion": (tn, fp, fn, tp),
        "FPR": fpr,
        "TPR": tpr,
    }


def roc_curve_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.linspace(0, 1, 101)
    tpr = []
    fpr = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.uint8)
        tn, fp, fn, tp = compute_confusion(y_true, y_pred)
        tpr.append(tp / (tp + fn) if (tp + fn) else 0.0)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)

    fpr = np.array(fpr[::-1])
    tpr = np.array(tpr[::-1])
    return fpr, tpr


def auc_from_curve(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def evaluate_subset(model, test_dir: Path, device, max_images: int = 9, eval_size: int = 128):
    sat_dir = test_dir / "sat"
    gt_dir = test_dir / "gt"
    image_files = sorted(sat_dir.glob("*_sat.jpg"))[:max_images]

    results = {
        "without_segmentation": {name: {"y_true": [], "y_prob": []} for name in MODEL_NAMES},
        "with_segmentation": {name: {"y_true": [], "y_prob": []} for name in MODEL_NAMES},
    }

    for image_path in image_files:
        base_name = image_path.stem.replace("_sat", "")
        mask_path = gt_dir / f"{base_name}_mask.png"
        if not mask_path.exists():
            continue

        image_tensor, original_size = preprocess_image(str(image_path))
        ground_truth = load_ground_truth(str(mask_path), original_size)
        gt_eval = cv2.resize(ground_truth.astype(np.float32), (eval_size, eval_size), interpolation=cv2.INTER_NEAREST)
        gt_flat = gt_eval.flatten().astype(np.uint8)

        preds_without = get_predictions_without_segmentation(model, image_tensor, device, original_size)
        preds_with = get_predictions_with_segmentation(model, image_tensor, device, original_size)

        for model_name in MODEL_NAMES:
            prob_without = cv2.resize(
                preds_without[model_name].astype(np.float32),
                (eval_size, eval_size),
                interpolation=cv2.INTER_LINEAR,
            ).flatten()
            prob_with = cv2.resize(
                preds_with[model_name].astype(np.float32),
                (eval_size, eval_size),
                interpolation=cv2.INTER_NEAREST,
            ).flatten()

            results["without_segmentation"][model_name]["y_true"].append(gt_flat)
            results["without_segmentation"][model_name]["y_prob"].append(prob_without)
            results["with_segmentation"][model_name]["y_true"].append(gt_flat)
            results["with_segmentation"][model_name]["y_prob"].append(prob_with)

    final = {}
    for config in results:
        final[config] = {}
        for model_name in MODEL_NAMES:
            y_true = np.concatenate(results[config][model_name]["y_true"]).astype(np.uint8)
            y_prob = np.concatenate(results[config][model_name]["y_prob"]).astype(np.float32)
            final[config][model_name] = compute_metrics(y_true, y_prob, threshold=0.5)
    return final


def plot_roc(metrics_report: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"SegNet": "#1f77b4", "DeepLabV3+": "#2ca02c", "Hybrid": "#d62728"}
    titles = {
        "without_segmentation": "ROC Curves - Without Post-Processing",
        "with_segmentation": "ROC Curves - With Post-Processing",
    }
    for ax, config in zip(axes, ["without_segmentation", "with_segmentation"]):
        for model_name in MODEL_NAMES:
            metrics = metrics_report[config][model_name]
            ax.plot(
                metrics["FPR"],
                metrics["TPR"],
                color=colors[model_name],
                linewidth=2.2,
                label=f"{model_name} (AUC={metrics['ROC-AUC']:.3f})",
            )
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5)
        ax.set_title(titles[config], fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusions(metrics_report: dict, out_dir: Path):
    titles = {
        "without_segmentation": "confusion_matrices_without_segmentation.png",
        "with_segmentation": "confusion_matrices_with_segmentation.png",
    }
    cmaps = {"without_segmentation": "Blues", "with_segmentation": "Greens"}
    for config in ["without_segmentation", "with_segmentation"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, model_name in zip(axes, MODEL_NAMES):
            tn, fp, fn, tp = metrics_report[config][model_name]["Confusion"]
            matrix = np.array([[tn, fp], [fn, tp]])
            im = ax.imshow(matrix, cmap=cmaps[config])
            ax.set_title(model_name, fontweight="bold")
            ax.set_xticks([0, 1], ["No Oil", "Oil"])
            ax.set_yticks([0, 1], ["No Oil", "Oil"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", fontsize=12, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(config.replace("_", " ").title(), fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / titles[config], dpi=300, bbox_inches="tight")
        plt.close()


def plot_metrics_comparison(metrics_report: dict, out_path: Path):
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "ROC-AUC"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    x = np.arange(len(MODEL_NAMES))
    width = 0.35
    for ax, metric_name in zip(axes, metric_names):
        without_vals = [metrics_report["without_segmentation"][m][metric_name] for m in MODEL_NAMES]
        with_vals = [metrics_report["with_segmentation"][m][metric_name] for m in MODEL_NAMES]
        ax.bar(x - width / 2, without_vals, width, label="Without Post", color="#87CEEB", edgecolor="black")
        ax.bar(x + width / 2, with_vals, width, label="With Post", color="#FF9999", edgecolor="black")
        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(x, MODEL_NAMES, rotation=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_heatmap(metrics_report: dict, out_path: Path):
    rows = []
    labels = []
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "IoU", "Dice", "ROC-AUC"]
    for config, label in [("without_segmentation", "Without Post"), ("with_segmentation", "With Post")]:
        for model_name in MODEL_NAMES:
            labels.append(f"{label} | {model_name}")
            rows.append([metrics_report[config][model_name][metric] for metric in metric_names])
    matrix = np.array(rows)
    plt.figure(figsize=(12, 5.5))
    plt.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.xticks(range(len(metric_names)), metric_names, rotation=20)
    plt.yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(label="Score")
    plt.title("Metrics Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metrics_tables(metrics_report: dict, out_dir: Path):
    csv_path = out_dir / "metrics_table.csv"
    headers = ["Configuration", "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "IoU", "Dice", "ROC-AUC"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for config, label in [("without_segmentation", "Without Post-Processing"), ("with_segmentation", "With Post-Processing")]:
            for model_name in MODEL_NAMES:
                metrics = metrics_report[config][model_name]
                writer.writerow([
                    label, model_name,
                    f"{metrics['Accuracy']:.6f}",
                    f"{metrics['Precision']:.6f}",
                    f"{metrics['Recall']:.6f}",
                    f"{metrics['F1-Score']:.6f}",
                    f"{metrics['Specificity']:.6f}",
                    f"{metrics['IoU']:.6f}",
                    f"{metrics['Dice']:.6f}",
                    f"{metrics['ROC-AUC']:.6f}",
                ])

    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.axis("off")
    rows = []
    for config, label in [("without_segmentation", "Without Post"), ("with_segmentation", "With Post")]:
        for model_name in MODEL_NAMES:
            metrics = metrics_report[config][model_name]
            rows.append([
                label, model_name,
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
        colLabels=["Config", "Model", "Acc", "Prec", "Recall", "F1", "Spec", "IoU", "Dice", "AUC"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    plt.title("Metrics Table", fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    txt_path = out_dir / "metrics_report.txt"
    with txt_path.open("w") as handle:
        handle.write("OIL SPILL DETECTION - IEEE RESULTS PACKAGE METRICS\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("Quantitative evaluation used a 9-image subset with 128x128 downsampled masks\n")
        handle.write("to keep CPU-side packaging stable while still using real trained-model outputs.\n\n")
        for config, label in [("without_segmentation", "WITHOUT Post-Processing"), ("with_segmentation", "WITH Post-Processing")]:
            handle.write(label + "\n")
            handle.write("-" * 80 + "\n")
            for model_name in MODEL_NAMES:
                metrics = metrics_report[config][model_name]
                handle.write(
                    f"{model_name}: Acc={metrics['Accuracy']:.4f}, Prec={metrics['Precision']:.4f}, "
                    f"Recall={metrics['Recall']:.4f}, F1={metrics['F1-Score']:.4f}, "
                    f"Spec={metrics['Specificity']:.4f}, IoU={metrics['IoU']:.4f}, "
                    f"Dice={metrics['Dice']:.4f}, AUC={metrics['ROC-AUC']:.4f}\n"
                )
            handle.write("\n")


def copy_assets(base_dir: Path, out_dir: Path):
    gradcam_src = base_dir / "results_gradcam" / "all_models_grid_best"
    gradcam_dst = out_dir / "gradcam_grids"
    gradcam_dst.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(gradcam_src.glob("*.png")):
        shutil.copy2(file_path, gradcam_dst / file_path.name)

    qual_src = base_dir / "results" / "all_three_models"
    qual_dst = out_dir / "qualitative_comparisons"
    qual_dst.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(qual_src.glob("*.png")):
        shutil.copy2(file_path, qual_dst / file_path.name)


def write_summary(out_dir: Path):
    summary = out_dir / "README_RESULTS_PACKAGE.txt"
    with summary.open("w") as handle:
        handle.write("IEEE Results Package\n")
        handle.write("=" * 50 + "\n")
        handle.write("This folder contains quantitative plots and Grad-CAM based qualitative\n")
        handle.write("figures for the oil spill detection minor project and its Grad-CAM\n")
        handle.write("major-project extension.\n\n")
        handle.write("Main files:\n")
        handle.write("- roc_curves_comparison.png\n")
        handle.write("- confusion_matrices_without_segmentation.png\n")
        handle.write("- confusion_matrices_with_segmentation.png\n")
        handle.write("- metrics_comparison_chart.png\n")
        handle.write("- metrics_heatmap.png\n")
        handle.write("- metrics_table.csv\n")
        handle.write("- metrics_table.png\n")
        handle.write("- metrics_report.txt\n")
        handle.write("- gradcam_grids/\n")
        handle.write("- qualitative_comparisons/\n")


def main():
    base_dir = Path(__file__).parent
    out_dir = base_dir / "results" / "ieee_results_package"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING LIGHTWEIGHT IEEE RESULTS PACKAGE")
    print("=" * 80)

    device = get_device()
    model = HybridOilSpillModel(num_classes=1, backbone="resnet50")
    checkpoint_path = base_dir / "outputs" / "hybrid" / "checkpoints" / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()

    metrics_report = evaluate_subset(
        model=model,
        test_dir=base_dir / "dataset" / "SOS_dataset" / "test" / "palsar",
        device=device,
        max_images=9,
        eval_size=128,
    )

    plot_roc(metrics_report, out_dir / "roc_curves_comparison.png")
    plot_confusions(metrics_report, out_dir)
    plot_metrics_comparison(metrics_report, out_dir / "metrics_comparison_chart.png")
    plot_metrics_heatmap(metrics_report, out_dir / "metrics_heatmap.png")
    save_metrics_tables(metrics_report, out_dir)
    copy_assets(base_dir, out_dir)
    write_summary(out_dir)

    print(f"Package ready at: {out_dir}")


if __name__ == "__main__":
    main()
