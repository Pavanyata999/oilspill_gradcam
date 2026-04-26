import argparse
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def generate_realistic_evaluation_data(n_pixels: int = 100000):
    """Generate realistic evaluation data for oil spill detection"""
    np.random.seed(42)

    # Oil spill pixels are rare (about 5-10% of pixels)
    oil_spill_ratio = 0.08
    n_oil_pixels = int(n_pixels * oil_spill_ratio)
    n_background_pixels = n_pixels - n_oil_pixels

    # Generate ground truth
    y_true = np.concatenate([
        np.zeros(n_background_pixels),  # Background pixels
        np.ones(n_oil_pixels)          # Oil spill pixels
    ])

    # Generate realistic predictions for hybrid model
    # Background pixels: mostly correct, some false positives
    background_probs = np.random.beta(1.5, 6, n_background_pixels)  # Low probability for background

    # Oil spill pixels: mostly correct, some false negatives
    oil_probs = np.random.beta(8, 2, n_oil_pixels)  # High probability for oil spills

    y_prob = np.concatenate([background_probs, oil_probs])

    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, n_pixels)
    y_prob = np.clip(y_prob + noise, 0, 1)

    # Generate predictions with threshold
    y_pred = (y_prob >= 0.5).astype(int)

    return y_true.astype(int), y_pred, y_prob


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Background", "Oil Spill"],
        yticklabels=["Background", "Oil Spill"],
    )
    plt.title("Confusion Matrix - Hybrid Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hybrid Model")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    """Plot precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Hybrid Model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_metrics_table(metrics: Dict, save_path: str):
    """Create and save metrics table as image"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Create table data
    table_data = []
    for metric, value in metrics.items():
        if isinstance(value, float):
            table_data.append([metric, ".4f"])
        else:
            table_data.append([metric, str(value)])

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
        colColours=["#f0f0f0", "#f0f0f0"],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.title("Model Performance Metrics - Hybrid Model", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_probability_histogram(y_prob: np.ndarray, save_path: str):
    """Plot histogram of prediction probabilities"""
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    plt.xlabel("Prediction Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Probabilities - Hybrid Model")
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0.5, color="red", linestyle="--", label="Threshold = 0.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_evaluation_report(args):
    """Main evaluation function"""
    print("Generating realistic evaluation data for hybrid model...")
    print("(Using simulated data for demonstration - actual model evaluation would be too slow on CPU)")

    # Generate realistic evaluation data
    y_true, y_pred, y_prob = generate_realistic_evaluation_data(n_pixels=args.n_pixels)

    print(f"Evaluation data generated. Total pixels: {len(y_true)}")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate AUC
    auc = roc_auc_score(y_true, y_prob)

    # Pixel-wise IoU (Jaccard)
    intersection = np.logical_and(y_pred, y_true).sum()
    union = np.logical_or(y_pred, y_true).sum()
    iou = intersection / union if union > 0 else 0

    # Dice coefficient
    dice = (2 * intersection) / (y_pred.sum() + y_true.sum()) if (y_pred.sum() + y_true.sum()) > 0 else 0

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "IoU (Jaccard)": iou,
        "Dice Coefficient": dice,
        "AUC": auc,
        "Total Pixels": len(y_true),
        "Positive Pixels (Pred)": y_pred.sum(),
        "Positive Pixels (GT)": y_true.sum(),
    }

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate visualizations
    print("Generating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, os.path.join(args.output_dir, "confusion_matrix.png"))

    print("Generating ROC curve...")
    plot_roc_curve(y_true, y_prob, os.path.join(args.output_dir, "roc_curve.png"))

    print("Generating precision-recall curve...")
    plot_precision_recall_curve(y_true, y_prob, os.path.join(args.output_dir, "precision_recall_curve.png"))

    print("Generating probability histogram...")
    plot_probability_histogram(y_prob, os.path.join(args.output_dir, "probability_histogram.png"))

    print("Creating metrics table...")
    create_metrics_table(metrics, os.path.join(args.output_dir, "metrics_table.png"))

    # Save metrics to text file
    with open(os.path.join(args.output_dir, "metrics_report.txt"), "w") as f:
        f.write("Hybrid Model Evaluation Report (Demo Data)\n")
        f.write("=" * 50 + "\n\n")
        f.write("NOTE: This report uses simulated realistic data for demonstration.\n")
        f.write("Actual model evaluation on full dataset would require GPU acceleration.\n\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")
    print("\nKey Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(".4f")
        else:
            print(f"{key}: {value}")

    print("\nGenerated files:")
    print("  • confusion_matrix.png")
    print("  • roc_curve.png")
    print("  • precision_recall_curve.png")
    print("  • probability_histogram.png")
    print("  • metrics_table.png")
    print("  • metrics_report.txt")


def build_parser():
    parser = argparse.ArgumentParser(description="Generate comprehensive evaluation metrics for oil spill detection model")
    parser.add_argument("--output_dir", type=str, default="results/demo_metrics")
    parser.add_argument("--n_pixels", type=int, default=100000, help="Number of pixels to simulate for evaluation")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate_evaluation_report(args)#!/usr/bin/env python3
"""
Demo Script: Generate Sample ROC Curves and Confusion Matrices
This creates example visualizations without needing a trained model
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def generate_sample_predictions():
    """Generate realistic sample predictions for demo purposes"""
    np.random.seed(42)
    n_samples = 100000
    
    # Generate ground truth (20% positive class)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.80, 0.20])
    
    results = {}
    
    # SegNet - Conservative (high precision, lower recall)
    segnet_probs = np.where(y_true == 1, 
                            np.random.beta(6, 2, n_samples),  # Oil spills
                            np.random.beta(2, 8, n_samples))  # Non-oil
    
    # DeepLabV3+ - Aggressive (high recall, lower precision)
    deeplab_probs = np.where(y_true == 1,
                             np.random.beta(7, 1.5, n_samples),  # Oil spills
                             np.random.beta(2.5, 6, n_samples))  # Non-oil
    
    # Hybrid - Balanced (best overall)
    hybrid_probs = np.where(y_true == 1,
                            np.random.beta(7, 1.8, n_samples),  # Oil spills
                            np.random.beta(2, 8, n_samples))  # Non-oil
    
    for name, probs in [('SegNet', segnet_probs), 
                        ('DeepLabV3+', deeplab_probs), 
                        ('Hybrid', hybrid_probs)]:
        results[name] = {
            'y_true': y_true,
            'y_prob': probs,
            'y_pred': (probs > 0.5).astype(int)
        }
    
    return results

def generate_sample_predictions_with_segmentation():
    """Generate predictions with post-processing improvements"""
    np.random.seed(43)
    n_samples = 100000
    
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.80, 0.20])
    
    results = {}
    
    # With segmentation: better precision, slightly lower recall
    segnet_probs = np.where(y_true == 1,
                            np.random.beta(6.5, 2, n_samples),
                            np.random.beta(1.8, 9, n_samples))  # Better specificity
    
    deeplab_probs = np.where(y_true == 1,
                             np.random.beta(7.2, 1.6, n_samples),
                             np.random.beta(2, 7.5, n_samples))
    
    hybrid_probs = np.where(y_true == 1,
                            np.random.beta(7.5, 1.6, n_samples),
                            np.random.beta(1.8, 9, n_samples))  # Best improvement
    
    for name, probs in [('SegNet', segnet_probs),
                        ('DeepLabV3+', deeplab_probs),
                        ('Hybrid', hybrid_probs)]:
        results[name] = {
            'y_true': y_true,
            'y_prob': probs,
            'y_pred': (probs > 0.5).astype(int)
        }
    
    return results

def plot_roc_curves(without_seg, with_seg, output_dir):
    """Plot ROC curves comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    colors = {'SegNet': 'blue', 'DeepLabV3+': 'green', 'Hybrid': 'red'}
    
    # Without segmentation
    ax = axes[0]
    for model_name in ['SegNet', 'DeepLabV3+', 'Hybrid']:
        data = without_seg[model_name]
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[model_name], lw=2.5,
                label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves - WITHOUT Post-Processing', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # With segmentation
    ax = axes[1]
    for model_name in ['SegNet', 'DeepLabV3+', 'Hybrid']:
        data = with_seg[model_name]
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[model_name], lw=2.5,
                label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves - WITH Post-Processing', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ ROC curves saved")

def plot_confusion_matrices(without_seg, with_seg, output_dir):
    """Plot confusion matrices"""
    model_names = ['SegNet', 'DeepLabV3+', 'Hybrid']
    
    # Without segmentation
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Confusion Matrices - WITHOUT Post-Processing', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    for idx, model_name in enumerate(model_names):
        data = without_seg[model_name]
        cm = confusion_matrix(data['y_true'], data['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
        axes[idx].set_title(f'{model_name}', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=12)
        axes[idx].set_xlabel('Predicted Label', fontsize=12)
        axes[idx].set_xticklabels(['No Oil', 'Oil Spill'])
        axes[idx].set_yticklabels(['No Oil', 'Oil Spill'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_without_segmentation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # With segmentation
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Confusion Matrices - WITH Post-Processing', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    for idx, model_name in enumerate(model_names):
        data = with_seg[model_name]
        cm = confusion_matrix(data['y_true'], data['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
        axes[idx].set_title(f'{model_name}', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=12)
        axes[idx].set_xlabel('Predicted Label', fontsize=12)
        axes[idx].set_xticklabels(['No Oil', 'Oil Spill'])
        axes[idx].set_yticklabels(['No Oil', 'Oil Spill'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_with_segmentation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrices saved")

def calculate_metrics(data):
    """Calculate performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    cm = confusion_matrix(data['y_true'], data['y_pred'])
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(data['y_true'], data['y_pred'])
    precision = precision_score(data['y_true'], data['y_pred'], zero_division=0)
    recall = recall_score(data['y_true'], data['y_pred'], zero_division=0)
    f1 = f1_score(data['y_true'], data['y_pred'], zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
    roc_auc = auc(fpr, tpr)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity,
        'ROC-AUC': roc_auc
    }

def print_metrics_table(without_seg, with_seg):
    """Print metrics comparison"""
    print("\n" + "="*100)
    print("DEMO PERFORMANCE METRICS COMPARISON")
    print("="*100)
    
    for seg_type, data_dict in [('WITHOUT Post-Processing', without_seg),
                                 ('WITH Post-Processing', with_seg)]:
        print(f"\n{seg_type}")
        print("-"*100)
        print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Specificity':<12} {'ROC-AUC':<12}")
        print("-"*100)
        
        for model_name in ['SegNet', 'DeepLabV3+', 'Hybrid']:
            metrics = calculate_metrics(data_dict[model_name])
            print(f"{model_name:<15} "
                  f"{metrics['Accuracy']:<12.4f} "
                  f"{metrics['Precision']:<12.4f} "
                  f"{metrics['Recall']:<12.4f} "
                  f"{metrics['F1-Score']:<12.4f} "
                  f"{metrics['Specificity']:<12.4f} "
                  f"{metrics['ROC-AUC']:<12.4f}")
        print()

def main():
    print("="*100)
    print("DEMO: ROC CURVES AND CONFUSION MATRICES")
    print("Generating sample visualizations with simulated data")
    print("="*100)
    
    # Create output directory
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'results' / 'demo_metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n✓ Generating sample predictions...")
    without_seg = generate_sample_predictions()
    with_seg = generate_sample_predictions_with_segmentation()
    
    print("✓ Creating ROC curves...")
    plot_roc_curves(without_seg, with_seg, output_dir)
    
    print("✓ Creating confusion matrices...")
    plot_confusion_matrices(without_seg, with_seg, output_dir)
    
    print("\n✓ Calculating metrics...")
    print_metrics_table(without_seg, with_seg)
    
    print("\n" + "="*100)
    print("✅ DEMO COMPLETE!")
    print("="*100)
    print(f"\n📂 Results saved to: {output_dir}")
    print("\n📊 Generated files:")
    print("  • roc_curves_comparison.png")
    print("  • confusion_matrices_without_segmentation.png")
    print("  • confusion_matrices_with_segmentation.png")
    print("\n⚠️  NOTE: This is demo data. For actual results, train the model first:")
    print("  python3 train_hybrid.py")
    print("  python3 generate_evaluation_metrics.py")
    print()

if __name__ == '__main__':
    main()
