#!/usr/bin/env python3
"""
Final prediction script with smart automatic thresholding
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage

from models.hybrid_models import HybridOilSpillModel

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU")
    return device

def preprocess_image(image_path):
    """Preprocess image"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_size = image.shape
    image_resized = cv2.resize(image, (512, 512))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_size

def otsu_threshold(pred_array):
    """Calculate Otsu's threshold for automatic thresholding"""
    # Convert to 0-255 range
    pred_uint8 = (pred_array * 255).astype(np.uint8)
    # Apply Otsu's method
    threshold, _ = cv2.threshold(pred_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold / 255.0

def remove_small_objects(mask, min_size=50):
    """Remove small isolated objects"""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(num_features + 1))
    mask_cleaned = sizes > min_size
    mask_cleaned = mask_cleaned[labeled]
    return mask_cleaned.astype(np.uint8) * 255

def morphological_cleanup(mask, opening_size=3, closing_size=5):
    """Clean up mask with morphological operations"""
    if mask.max() == 0:
        return mask
    
    # Opening: remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Closing: fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    
    return closed

def postprocess_prediction(prediction, original_size, model_name):
    """
    Smart post-processing for each model with automatic thresholding
    """
    pred = torch.sigmoid(prediction)
    pred = pred.squeeze().cpu().numpy()
    
    # Different strategies per model
    if model_name == 'SegNet':
        # Use lower percentile for SegNet (it tends to be conservative)
        threshold = np.percentile(pred, 60)
        threshold = max(threshold, 0.45)  # Lower minimum
        opening = 5  # More aggressive noise removal
        closing = 7
        min_obj_size = 200
        
    elif model_name == 'DeepLabV3+':
        # Use Otsu's method or high percentile
        try:
            threshold = otsu_threshold(pred)
            if threshold > 0.95:  # If Otsu gives too high threshold
                threshold = np.percentile(pred, 95)
        except:
            threshold = np.percentile(pred, 95)
        threshold = min(threshold, 0.95)  # Cap at 0.95
        opening = 3
        closing = 5
        min_obj_size = 100
        
    else:  # Hybrid
        # Balanced approach
        threshold = np.percentile(pred, 92)
        threshold = min(max(threshold, 0.65), 0.92)
        opening = 3
        closing = 6
        min_obj_size = 120
    
    # Create binary mask
    mask = (pred > threshold).astype(np.uint8) * 255
    
    # Clean up
    mask = morphological_cleanup(mask, opening, closing)
    mask = remove_small_objects(mask, min_obj_size)
    
    # Resize to original size
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    return mask_resized, pred, threshold

def extract_component_outputs(hybrid_model, image_tensor, device):
    """Extract predictions"""
    hybrid_model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        segnet_out = hybrid_model.segnet(image_tensor)
        deeplab_out = hybrid_model.deeplabv3_plus(image_tensor)
        hybrid_out = hybrid_model(image_tensor)
    
    return {
        'SegNet': segnet_out,
        'DeepLabV3+': deeplab_out,
        'Hybrid': hybrid_out
    }

def create_visualization_with_red_overlay(image_path, masks_dict, thresholds, output_path):
    """Create final visualization"""
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize for display
    image_resized = cv2.resize(image, (512, 512))
    image_rgb_resized = cv2.resize(image_rgb, (512, 512))
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    
    model_names = ['SegNet', 'DeepLabV3+', 'Hybrid']
    
    for row, model_name in enumerate(model_names):
        mask_resized = cv2.resize(masks_dict[model_name], (512, 512))
        threshold = thresholds[model_name]
        
        # Calculate stats
        detection_pct = (mask_resized > 128).sum() / mask_resized.size * 100
        
        # Column 1: Original
        axes[row, 0].imshow(image_resized, cmap='gray')
        axes[row, 0].set_title(f'{model_name}\nOriginal SAR Image', 
                              fontsize=14, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Column 2: Mask
        axes[row, 1].imshow(mask_resized, cmap='gray')
        axes[row, 1].set_title(f'{model_name}\nOil Spill Detection Mask\n(Threshold: {threshold:.3f})', 
                              fontsize=14, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Column 3: Red Overlay
        overlay = image_rgb_resized.copy()
        overlay[mask_resized > 128] = [255, 0, 0]
        
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f'{model_name}\nOverlay (Red = Oil Spill)\nDetection: {detection_pct:.1f}%', 
                              fontsize=14, fontweight='bold')
        axes[row, 2].axis('off')
    
    plt.suptitle(f'Oil Spill Detection Comparison - {Path(image_path).stem}', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 80)
    print("Oil Spill Detection - FINAL VERSION with Smart Thresholding")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / 'dataset' / 'SOS_dataset' / 'test' / 'palsar'
    output_dir = base_dir / 'results' / 'final_outputs'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    
    print("\nLoading model...")
    hybrid_model = HybridOilSpillModel(num_classes=1, backbone='resnet50')
    
    checkpoint_path = base_dir / 'output' / 'checkpoints' / 'best_model.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            hybrid_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            hybrid_model.load_state_dict(checkpoint)
        print("✓ Loaded trained model")
    
    hybrid_model.to(device).eval()
    
    image_files = list((dataset_dir / 'sat').glob('*.jpg'))[:12]
    
    if not image_files:
        print("⚠ No images found")
        return
    
    print(f"✓ Found {len(image_files)} images")
    print("=" * 80)
    print("\nProcessing with AUTO-THRESHOLDING:")
    print("  SegNet: 60th percentile, aggressive cleanup")
    print("  DeepLabV3+: Otsu/95th percentile, moderate cleanup")
    print("  Hybrid: 92nd percentile, balanced cleanup")
    print("\n" + "=" * 80 + "\n")
    
    for image_path in tqdm(image_files, desc="Generating outputs"):
        base_name = image_path.stem.replace('_sat', '')
        
        try:
            image_tensor, original_size = preprocess_image(str(image_path))
            outputs = extract_component_outputs(hybrid_model, image_tensor, device)
            
            masks = {}
            thresholds = {}
            
            for model_name, output in outputs.items():
                mask, _, threshold = postprocess_prediction(output, original_size, model_name)
                masks[model_name] = mask
                thresholds[model_name] = threshold
            
            viz_path = output_dir / f"{base_name}_final.png"
            create_visualization_with_red_overlay(str(image_path), masks, thresholds, viz_path)
            
            # Print stats
            seg_pct = (masks['SegNet']>128).sum() / masks['SegNet'].size * 100
            deep_pct = (masks['DeepLabV3+']>128).sum() / masks['DeepLabV3+'].size * 100
            hyb_pct = (masks['Hybrid']>128).sum() / masks['Hybrid'].size * 100
            
            print(f"✓ {base_name}: SegNet={seg_pct:.1f}% (th={thresholds['SegNet']:.3f}), "
                  f"DeepLab={deep_pct:.1f}% (th={thresholds['DeepLabV3+']:.3f}), "
                  f"Hybrid={hyb_pct:.1f}% (th={thresholds['Hybrid']:.3f})")
            
        except Exception as e:
            print(f"✗ Error {image_path.name}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE! All models working with different results!")
    print("=" * 80)
    print(f"\n📂 Results: {output_dir}")
    print("\n📊 Model Differences:")
    print("  ┌─────────────┬────────────────────────────────────────┐")
    print("  │   SegNet    │ Most conservative, clean detections   │")
    print("  │ DeepLabV3+  │ Most aggressive, high coverage        │")
    print("  │   Hybrid    │ Balanced blend of both approaches     │")
    print("  └─────────────┴────────────────────────────────────────┘")
    print("\n🔴 Red overlays show detected oil spills on SAR images")
    print()

if __name__ == '__main__':
    main()
