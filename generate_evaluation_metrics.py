#!/usr/bin/env python3
"""
Evaluation Script: ROC Curves and Confusion Matrices
Compares performance with and without segmentation
"""

import os
import torch
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import seaborn as sns
from models.hybrid_models import HybridOilSpillModel
import warnings
warnings.filterwarnings('ignore')

def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
    return device

def preprocess_image(image_path):
    """Preprocess image for model input"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_size = image.shape
    image_resized = cv2.resize(image, (512, 512))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_size

def load_ground_truth(mask_path, original_size):
    """Load and preprocess ground truth mask"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    # Resize to match original size
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Convert to binary
    mask_binary = (mask_resized > 128).astype(np.float32)
    return mask_binary

def get_predictions_without_segmentation(model, image_tensor, device, original_size):
    """Get raw predictions without any post-processing"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Get predictions from all models
        segnet_out = model.segnet(image_tensor)
        deeplab_out = model.deeplabv3_plus(image_tensor)
        hybrid_out = model(image_tensor)
    
    # Convert to probabilities
    results = {}
    for name, output in [('SegNet', segnet_out), 
                         ('DeepLabV3+', deeplab_out), 
                         ('Hybrid', hybrid_out)]:
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_prob_resized = cv2.resize(pred_prob, (original_size[1], original_size[0]))
        results[name] = pred_prob_resized
    
    return results

def get_predictions_with_segmentation(model, image_tensor, device, original_size):
    """Get predictions with advanced post-processing"""
    from scipy import ndimage
    
    def remove_small_objects(mask, min_size=50):
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask
        sizes = ndimage.sum(mask, labeled, range(num_features + 1))
        mask_cleaned = sizes > min_size
        mask_cleaned = mask_cleaned[labeled]
        return mask_cleaned.astype(np.uint8) * 255
    
    def morphological_cleanup(mask, opening_size=3, closing_size=5):
        if mask.max() == 0:
            return mask
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        return closed
    
    # Get raw predictions first
    raw_preds = get_predictions_without_segmentation(model, image_tensor, device, (512, 512))
    
    results = {}
    for model_name, pred in raw_preds.items():
        # Apply model-specific thresholds
        if model_name == 'SegNet':
            threshold = max(np.percentile(pred, 60), 0.45)
            opening, closing, min_obj = 5, 7, 200
        elif model_name == 'DeepLabV3+':
            threshold = min(np.percentile(pred, 95), 0.95)
            opening, closing, min_obj = 3, 5, 100
        else:  # Hybrid
            threshold = np.clip(np.percentile(pred, 92), 0.65, 0.92)
            opening, closing, min_obj = 3, 6, 120
        
        # Create binary mask
        mask = (pred > threshold).astype(np.uint8) * 255
        
        # Apply morphological operations
        mask = morphological_cleanup(mask, opening, closing)
        mask = remove_small_objects(mask, min_obj)
        
        # Resize to original size
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Convert to probability-like values (0 or 1)
        results[model_name] = (mask_resized / 255.0).astype(np.float32)
    
    return results

def evaluate_all_models(model, test_dir, device, output_dir, max_images=36, eval_size=128):
    """Evaluate all models with and without segmentation"""
    
    sat_dir = test_dir / 'sat'
    gt_dir = test_dir / 'gt'
    
    # Get all test images
    image_files = sorted(list(sat_dir.glob('*_sat.jpg')))[:max_images]
    
    if not image_files:
        print("⚠ No test images found")
        return None
    
    print(f"\n✓ Found {len(image_files)} test images")
    
    # Storage for predictions and ground truth
    all_results = {
        'without_segmentation': {
            'SegNet': {'y_true': [], 'y_pred': [], 'y_prob': []},
            'DeepLabV3+': {'y_true': [], 'y_pred': [], 'y_prob': []},
            'Hybrid': {'y_true': [], 'y_pred': [], 'y_prob': []}
        },
        'with_segmentation': {
            'SegNet': {'y_true': [], 'y_pred': [], 'y_prob': []},
            'DeepLabV3+': {'y_true': [], 'y_pred': [], 'y_prob': []},
            'Hybrid': {'y_true': [], 'y_pred': [], 'y_prob': []}
        }
    }
    
    print("\nProcessing test images...")
    for image_path in tqdm(image_files, desc="Evaluating"):
        base_name = image_path.stem.replace('_sat', '')
        mask_path = gt_dir / f'{base_name}_mask.png'
        
        if not mask_path.exists():
            continue
        
        try:
            # Preprocess
            image_tensor, original_size = preprocess_image(str(image_path))
            ground_truth = load_ground_truth(str(mask_path), original_size)
            
            # Get predictions without segmentation
            preds_without = get_predictions_without_segmentation(
                model, image_tensor, device, original_size
            )
            
            # Get predictions with segmentation
            preds_with = get_predictions_with_segmentation(
                model, image_tensor, device, original_size
            )
            
            # Downsample before flattening to keep evaluation memory manageable on CPU
            gt_eval = cv2.resize(
                ground_truth.astype(np.float32),
                (eval_size, eval_size),
                interpolation=cv2.INTER_NEAREST,
            )
            gt_flat = gt_eval.flatten()
            
            # Store results for each model
            for model_name in ['SegNet', 'DeepLabV3+', 'Hybrid']:
                # Without segmentation
                pred_prob_without = cv2.resize(
                    preds_without[model_name].astype(np.float32),
                    (eval_size, eval_size),
                    interpolation=cv2.INTER_LINEAR,
                ).flatten()
                pred_binary_without = (pred_prob_without > 0.5).astype(int)
                
                all_results['without_segmentation'][model_name]['y_true'].extend(gt_flat)
                all_results['without_segmentation'][model_name]['y_pred'].extend(pred_binary_without)
                all_results['without_segmentation'][model_name]['y_prob'].extend(pred_prob_without)
                
                # With segmentation
                pred_prob_with = cv2.resize(
                    preds_with[model_name].astype(np.float32),
                    (eval_size, eval_size),
                    interpolation=cv2.INTER_NEAREST,
                ).flatten()
                pred_binary_with = (pred_prob_with > 0.5).astype(int)
                
                all_results['with_segmentation'][model_name]['y_true'].extend(gt_flat)
                all_results['with_segmentation'][model_name]['y_pred'].extend(pred_binary_with)
                all_results['with_segmentation'][model_name]['y_prob'].extend(pred_prob_with)
                
        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {e}")
            continue

    return all_results

def plot_roc_curves(all_results, output_dir):
    """Plot ROC curves for all models with and without segmentation"""
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    colors = {'SegNet': 'blue', 'DeepLabV3+': 'green', 'Hybrid': 'red'}
    
    # Plot without segmentation
    ax = axes[0]
    for model_name in ['SegNet', 'DeepLabV3+', 'Hybrid']:
        data = all_results['without_segmentation'][model_name]
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
    
    # Plot with segmentation
    ax = axes[1]
    for model_name in ['SegNet', 'DeepLabV3+', 'Hybrid']:
        data = all_results['with_segmentation'][model_name]
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
    print("\n✓ ROC curves saved")

def plot_confusion_matrices(all_results, output_dir):
    """Plot confusion matrices for all models"""
    
    model_names = ['SegNet', 'DeepLabV3+', 'Hybrid']
    
    # Without segmentation
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Confusion Matrices - WITHOUT Post-Processing', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    for idx, model_name in enumerate(model_names):
        data = all_results['without_segmentation'][model_name]
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
        data = all_results['with_segmentation'][model_name]
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

def calculate_detailed_metrics(all_results):
    """Calculate detailed metrics for all models"""
    
    metrics_report = {}
    
    for seg_type in ['without_segmentation', 'with_segmentation']:
        metrics_report[seg_type] = {}
        
        for model_name in ['SegNet', 'DeepLabV3+', 'Hybrid']:
            data = all_results[seg_type][model_name]
            
            # Calculate metrics
            accuracy = accuracy_score(data['y_true'], data['y_pred'])
            precision = precision_score(data['y_true'], data['y_pred'], zero_division=0)
            recall = recall_score(data['y_true'], data['y_pred'], zero_division=0)
            f1 = f1_score(data['y_true'], data['y_pred'], zero_division=0)
            
            # Calculate ROC AUC
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
            roc_auc = auc(fpr, tpr)
            
            # Calculate specificity
            cm = confusion_matrix(data['y_true'], data['y_pred'])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                specificity = 0
            
            metrics_report[seg_type][model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Specificity': specificity,
                'ROC-AUC': roc_auc
            }
    
    return metrics_report

def print_metrics_table(metrics_report):
    """Print metrics in a formatted table"""
    
    print("\n" + "="*100)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*100)
    
    for seg_type in ['without_segmentation', 'with_segmentation']:
        title = "WITHOUT Post-Processing" if seg_type == 'without_segmentation' else "WITH Post-Processing"
        print(f"\n{title}")
        print("-"*100)
        print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Specificity':<12} {'ROC-AUC':<12}")
        print("-"*100)
        
        for model_name in ['SegNet', 'DeepLabV3+', 'Hybrid']:
            metrics = metrics_report[seg_type][model_name]
            print(f"{model_name:<15} "
                  f"{metrics['Accuracy']:<12.4f} "
                  f"{metrics['Precision']:<12.4f} "
                  f"{metrics['Recall']:<12.4f} "
                  f"{metrics['F1-Score']:<12.4f} "
                  f"{metrics['Specificity']:<12.4f} "
                  f"{metrics['ROC-AUC']:<12.4f}")
        print()

def save_metrics_comparison_plot(metrics_report, output_dir):
    """Create a visual comparison of metrics"""
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC']
    model_names = ['SegNet', 'DeepLabV3+', 'Hybrid']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Metrics Comparison: With vs Without Post-Processing', 
                 fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        without_vals = [metrics_report['without_segmentation'][m][metric_name] 
                       for m in model_names]
        with_vals = [metrics_report['with_segmentation'][m][metric_name] 
                    for m in model_names]
        
        bars1 = ax.bar(x - width/2, without_vals, width, label='Without Post-Processing',
                       color='skyblue', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, with_vals, width, label='With Post-Processing',
                       color='lightcoral', edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Metrics comparison chart saved")

def main():
    print("="*100)
    print("OIL SPILL DETECTION - COMPREHENSIVE EVALUATION")
    print("Generating ROC Curves, Confusion Matrices, and Performance Metrics")
    print("="*100)
    
    base_dir = Path(__file__).parent
    test_dir = base_dir / 'dataset' / 'SOS_dataset' / 'test' / 'palsar'
    output_dir = base_dir / 'results' / 'evaluation_metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    
    # Load model
    print("\nLoading trained model...")
    hybrid_model = HybridOilSpillModel(num_classes=1, backbone='resnet50')
    checkpoint_path = base_dir / 'outputs' / 'hybrid' / 'checkpoints' / 'best_model.pth'
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            hybrid_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            hybrid_model.load_state_dict(checkpoint)
        print("✓ Model loaded successfully")
    else:
        print("⚠ No trained model found, using untrained model")
    
    hybrid_model.to(device).eval()
    
    # Evaluate models
    all_results = evaluate_all_models(hybrid_model, test_dir, device, output_dir)
    
    if all_results is None:
        return
    
    print("\n" + "="*100)
    print("Generating visualizations...")
    print("="*100)
    
    # Generate visualizations
    plot_roc_curves(all_results, output_dir)
    plot_confusion_matrices(all_results, output_dir)
    
    # Calculate metrics
    metrics_report = calculate_detailed_metrics(all_results)
    
    # Print metrics
    print_metrics_table(metrics_report)
    
    # Save comparison chart
    save_metrics_comparison_plot(metrics_report, output_dir)
    
    # Save metrics to file
    metrics_file = output_dir / 'metrics_report.txt'
    with open(metrics_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("OIL SPILL DETECTION - PERFORMANCE METRICS REPORT\n")
        f.write("="*100 + "\n\n")
        
        for seg_type in ['without_segmentation', 'with_segmentation']:
            title = "WITHOUT Post-Processing" if seg_type == 'without_segmentation' else "WITH Post-Processing"
            f.write(f"\n{title}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Specificity':<12} {'ROC-AUC':<12}\n")
            f.write("-"*100 + "\n")
            
            for model_name in ['SegNet', 'DeepLabV3+', 'Hybrid']:
                metrics = metrics_report[seg_type][model_name]
                f.write(f"{model_name:<15} "
                       f"{metrics['Accuracy']:<12.4f} "
                       f"{metrics['Precision']:<12.4f} "
                       f"{metrics['Recall']:<12.4f} "
                       f"{metrics['F1-Score']:<12.4f} "
                       f"{metrics['Specificity']:<12.4f} "
                       f"{metrics['ROC-AUC']:<12.4f}\n")
            f.write("\n")
    
    print("\n" + "="*100)
    print("✅ EVALUATION COMPLETE!")
    print("="*100)
    print(f"\n📂 Results saved to: {output_dir}")
    print("\n📊 Generated files:")
    print("  • roc_curves_comparison.png")
    print("  • confusion_matrices_without_segmentation.png")
    print("  • confusion_matrices_with_segmentation.png")
    print("  • metrics_comparison_chart.png")
    print("  • metrics_report.txt")
    print()

if __name__ == '__main__':
    main()
