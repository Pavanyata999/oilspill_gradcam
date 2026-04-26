#!/usr/bin/env python3
"""
Generate a single IEEE-style results package folder containing:
- Quantitative evaluation plots
- Metrics report and tables
- Metrics heatmap
- Best Grad-CAM comparison grids
- Best qualitative comparison figures
"""

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import generate_evaluation_metrics as gem


def save_metrics_csv(metrics_report, output_dir: Path) -> Path:
    csv_path = output_dir / "metrics_table.csv"
    fieldnames = [
        "Configuration",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Specificity",
        "ROC-AUC",
    ]

    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for seg_type, label in [
            ("without_segmentation", "Without Post-Processing"),
            ("with_segmentation", "With Post-Processing"),
        ]:
            for model_name in ["SegNet", "DeepLabV3+", "Hybrid"]:
                metrics = metrics_report[seg_type][model_name]
                writer.writerow(
                    {
                        "Configuration": label,
                        "Model": model_name,
                        "Accuracy": f"{metrics['Accuracy']:.6f}",
                        "Precision": f"{metrics['Precision']:.6f}",
                        "Recall": f"{metrics['Recall']:.6f}",
                        "F1-Score": f"{metrics['F1-Score']:.6f}",
                        "Specificity": f"{metrics['Specificity']:.6f}",
                        "ROC-AUC": f"{metrics['ROC-AUC']:.6f}",
                    }
                )
    return csv_path


def save_metrics_table_png(metrics_report, output_dir: Path) -> Path:
    rows = []
    for seg_type, label in [
        ("without_segmentation", "Without Post"),
        ("with_segmentation", "With Post"),
    ]:
        for model_name in ["SegNet", "DeepLabV3+", "Hybrid"]:
            metrics = metrics_report[seg_type][model_name]
            rows.append(
                [
                    label,
                    model_name,
                    f"{metrics['Accuracy']:.4f}",
                    f"{metrics['Precision']:.4f}",
                    f"{metrics['Recall']:.4f}",
                    f"{metrics['F1-Score']:.4f}",
                    f"{metrics['Specificity']:.4f}",
                    f"{metrics['ROC-AUC']:.4f}",
                ]
            )

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Config",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Specificity",
            "ROC-AUC",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.6)
    plt.title("Performance Metrics Table", fontsize=16, fontweight="bold", pad=14)
    out_path = output_dir / "metrics_table.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def save_metrics_heatmap(metrics_report, output_dir: Path) -> Path:
    row_labels = []
    data = []
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "ROC-AUC"]
    for seg_type, label in [
        ("without_segmentation", "Without Post"),
        ("with_segmentation", "With Post"),
    ]:
        for model_name in ["SegNet", "DeepLabV3+", "Hybrid"]:
            row_labels.append(f"{label} | {model_name}")
            metrics = metrics_report[seg_type][model_name]
            data.append([metrics[name] for name in metric_names])

    matrix = np.array(data)
    plt.figure(figsize=(11, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=metric_names,
        yticklabels=row_labels,
        cbar_kws={"label": "Score"},
    )
    plt.title("Metrics Heatmap for All Models", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "metrics_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def copy_best_figures(base_dir: Path, output_dir: Path) -> list[Path]:
    copied = []

    gradcam_src = base_dir / "results_gradcam" / "all_models_grid_best"
    gradcam_dst = output_dir / "gradcam_grids"
    gradcam_dst.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(gradcam_src.glob("*.png")):
        destination = gradcam_dst / file_path.name
        shutil.copy2(file_path, destination)
        copied.append(destination)

    qualitative_src = base_dir / "results" / "all_three_models"
    qualitative_dst = output_dir / "qualitative_comparisons"
    qualitative_dst.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(qualitative_src.glob("*.png")):
        destination = qualitative_dst / file_path.name
        shutil.copy2(file_path, destination)
        copied.append(destination)

    return copied


def save_summary(metrics_report, output_dir: Path, evaluation_dir: Path) -> Path:
    summary_path = output_dir / "README_RESULTS_PACKAGE.txt"
    with summary_path.open("w") as handle:
        handle.write("IEEE Results Package - Oil Spill Detection\n")
        handle.write("=" * 60 + "\n\n")
        handle.write("This folder contains the final quantitative and qualitative outputs\n")
        handle.write("for the minor project paper and the Grad-CAM extension for the\n")
        handle.write("major-project direction.\n\n")
        handle.write("Included files:\n")
        handle.write("- roc_curves_comparison.png\n")
        handle.write("- confusion_matrices_without_segmentation.png\n")
        handle.write("- confusion_matrices_with_segmentation.png\n")
        handle.write("- metrics_comparison_chart.png\n")
        handle.write("- metrics_table.csv\n")
        handle.write("- metrics_table.png\n")
        handle.write("- metrics_heatmap.png\n")
        handle.write("- metrics_report.txt\n")
        handle.write("- gradcam_grids/\n")
        handle.write("- qualitative_comparisons/\n\n")
        handle.write("Best scores with post-processing:\n")
        for model_name in ["SegNet", "DeepLabV3+", "Hybrid"]:
            metrics = metrics_report["with_segmentation"][model_name]
            handle.write(
                f"- {model_name}: "
                f"Acc={metrics['Accuracy']:.4f}, "
                f"Prec={metrics['Precision']:.4f}, "
                f"Recall={metrics['Recall']:.4f}, "
                f"F1={metrics['F1-Score']:.4f}, "
                f"AUC={metrics['ROC-AUC']:.4f}\n"
            )
        handle.write(f"\nSource evaluation folder: {evaluation_dir}\n")
    return summary_path


def main():
    base_dir = Path(__file__).parent
    package_dir = base_dir / "results" / "ieee_results_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    max_images = 12

    print("=" * 100)
    print("GENERATING IEEE RESULTS PACKAGE")
    print("=" * 100)

    device = gem.get_device()
    test_dir = base_dir / "dataset" / "SOS_dataset" / "test" / "palsar"
    evaluation_dir = base_dir / "results" / "evaluation_metrics"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading trained hybrid checkpoint...")
    hybrid_model = gem.HybridOilSpillModel(num_classes=1, backbone="resnet50")
    checkpoint_path = base_dir / "outputs" / "hybrid" / "checkpoints" / "best_model.pth"
    checkpoint = gem.torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        hybrid_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        hybrid_model.load_state_dict(checkpoint)
    hybrid_model.to(device).eval()
    print("Checkpoint loaded.")

    print("\nRunning quantitative evaluation...")
    all_results = gem.evaluate_all_models(
        hybrid_model,
        test_dir,
        device,
        evaluation_dir,
        max_images=max_images,
        eval_size=128,
    )
    gem.plot_roc_curves(all_results, evaluation_dir)
    gem.plot_confusion_matrices(all_results, evaluation_dir)
    metrics_report = gem.calculate_detailed_metrics(all_results)
    gem.print_metrics_table(metrics_report)
    gem.save_metrics_comparison_plot(metrics_report, evaluation_dir)

    metrics_report_txt = evaluation_dir / "metrics_report.txt"
    with metrics_report_txt.open("w") as handle:
        handle.write("=" * 100 + "\n")
        handle.write("OIL SPILL DETECTION - PERFORMANCE METRICS REPORT\n")
        handle.write("=" * 100 + "\n\n")
        for seg_type, label in [
            ("without_segmentation", "WITHOUT Post-Processing"),
            ("with_segmentation", "WITH Post-Processing"),
        ]:
            handle.write(label + "\n")
            handle.write("-" * 100 + "\n")
            handle.write(
                f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} "
                f"{'F1-Score':<12} {'Specificity':<12} {'ROC-AUC':<12}\n"
            )
            handle.write("-" * 100 + "\n")
            for model_name in ["SegNet", "DeepLabV3+", "Hybrid"]:
                metrics = metrics_report[seg_type][model_name]
                handle.write(
                    f"{model_name:<15} {metrics['Accuracy']:<12.4f} {metrics['Precision']:<12.4f} "
                    f"{metrics['Recall']:<12.4f} {metrics['F1-Score']:<12.4f} "
                    f"{metrics['Specificity']:<12.4f} {metrics['ROC-AUC']:<12.4f}\n"
                )
            handle.write("\n")

    print("\nCreating paper-ready tables and heatmap...")
    save_metrics_csv(metrics_report, package_dir)
    save_metrics_table_png(metrics_report, package_dir)
    save_metrics_heatmap(metrics_report, package_dir)

    print("Copying core evaluation figures...")
    for file_name in [
        "roc_curves_comparison.png",
        "confusion_matrices_without_segmentation.png",
        "confusion_matrices_with_segmentation.png",
        "metrics_comparison_chart.png",
        "metrics_report.txt",
    ]:
        shutil.copy2(evaluation_dir / file_name, package_dir / file_name)

    print("Copying Grad-CAM grids and qualitative comparisons...")
    copy_best_figures(base_dir, package_dir)

    print("Writing package summary...")
    save_summary(metrics_report, package_dir, evaluation_dir)

    print(f"\nEvaluation subset size: {max_images} test images")
    print("\n" + "=" * 100)
    print("IEEE RESULTS PACKAGE READY")
    print("=" * 100)
    print(f"Saved to: {package_dir}")


if __name__ == "__main__":
    main()
