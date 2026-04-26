import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from data_loader import collect_image_mask_pairs
from models.deeplabv3_model import DeepLabV3Plus
from models.hybrid_models import HybridOilSpillModel
from models.segnet_model import SegNet
from utils import get_device, load_model_checkpoint, postprocess_prediction, preprocess_image


class BinarySegmentationTarget:
    def __init__(self, mask: np.ndarray):
        self.mask = torch.from_numpy(mask.astype(np.float32))

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        output = torch.sigmoid(model_output[0, 0])
        mask = self.mask.to(model_output.device)
        return (output * mask).sum()


def create_model(model_type: str, backbone: str, pretrained: bool = False):
    if model_type == "deeplabv3_plus":
        return DeepLabV3Plus(num_classes=1, backbone=backbone, pretrained=pretrained)
    if model_type == "segnet":
        return SegNet(num_classes=1, in_channels=3)
    if model_type == "hybrid":
        return HybridOilSpillModel(num_classes=1, backbone=backbone, pretrained=pretrained)
    raise ValueError(f"Unknown model type: {model_type}")


def load_mask(mask_path: Optional[str], output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    if not mask_path:
        return np.zeros((height, width), dtype=np.uint8)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)
    mask = (mask > 0).astype(np.uint8) * 255
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def compute_gradcam(model, input_tensor: torch.Tensor, threshold: float) -> np.ndarray:
    target_layer = model.get_gradcam_target_layer()
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        target_mask = (probabilities >= threshold).astype(np.float32)
        if target_mask.sum() == 0:
            target_mask = probabilities.astype(np.float32)

        target = BinarySegmentationTarget(target_mask)(logits)
        target.backward()

        activation_map = activations[0][0]
        gradient_map = gradients[0][0]
        weights = gradient_map.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activation_map).sum(dim=0)).cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam
    finally:
        forward_handle.remove()
        backward_handle.remove()


def create_gradcam_overlay(original_image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    height, width = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_image, 0.55, heatmap, 0.45, 0)
    return overlay


def panel_image(image: np.ndarray, label: str, panel_size: tuple[int, int]) -> np.ndarray:
    panel_width, panel_height = panel_size
    label_height = 42
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    resized = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((panel_height + label_height, panel_width, 3), 255, dtype=np.uint8)
    canvas[label_height:, :, :] = resized
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def create_grid(sample_id: str, original_image: np.ndarray, ground_truth: np.ndarray, model_outputs: list[dict]) -> np.ndarray:
    panel_size = (300, 300)
    spacer = 10
    row_label_width = 150
    title_height = 60

    rows = []
    for model_output in model_outputs:
        row_label = np.full((panel_size[1] + 42, row_label_width, 3), 245, dtype=np.uint8)
        cv2.putText(row_label, model_output["name"], (12, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

        panels = [
            panel_image(original_image, "Original", panel_size),
            panel_image(ground_truth, "Ground Truth", panel_size),
            panel_image(model_output["prediction"], "Prediction", panel_size),
            panel_image(model_output["gradcam"], "Grad-CAM", panel_size),
        ]

        row = row_label
        for panel in panels:
            separator = np.full((row.shape[0], spacer, 3), 255, dtype=np.uint8)
            row = np.hstack([row, separator, panel])
        rows.append(row)

    grid = rows[0]
    for row in rows[1:]:
        separator = np.full((spacer, row.shape[1], 3), 255, dtype=np.uint8)
        grid = np.vstack([grid, separator, row])

    title = np.full((title_height, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        title,
        f"Oil Spill Detection Comparison with Grad-CAM - Image {sample_id}",
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([title, grid])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all-model Grad-CAM comparison grids")
    parser.add_argument("--data_dir", type=str, default="dataset/SOS_dataset/test")
    parser.add_argument("--satellite_type", type=str, default="palsar", choices=["both", "palsar", "sentinel"])
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=9)
    parser.add_argument("--output_dir", type=str, default="results_gradcam/all_models_grid")
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ("SegNet", "segnet", Path("outputs/segnet/checkpoints/best_model.pth")),
        ("DeepLabV3+", "deeplabv3_plus", Path("outputs/deeplabv3_plus/checkpoints/best_model.pth")),
        ("Hybrid", "hybrid", Path("outputs/hybrid/checkpoints/best_model.pth")),
    ]

    models = []
    for display_name, model_type, checkpoint_path in model_specs:
        model = create_model(model_type=model_type, backbone=args.backbone, pretrained=False).to(device)
        checkpoint = load_model_checkpoint(model, str(checkpoint_path), device=device)
        model_image_size = checkpoint.get("image_size", 256)
        model.eval()
        models.append(
            {
                "display_name": display_name,
                "model_type": model_type,
                "model": model,
                "image_size": model_image_size,
            }
        )
    samples = collect_image_mask_pairs(args.data_dir, satellite_type=args.satellite_type, require_masks=False)[: args.limit]

    for index, sample in enumerate(samples, start=1):
        original_image_bgr = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
        if original_image_bgr is None:
            print(f"Skipping unreadable image: {sample.image_path}")
            continue
        original_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
        ground_truth = load_mask(sample.mask_path, (original_image.shape[1], original_image.shape[0]))

        rows = []
        for model_info in models:
            image_tensor, _ = preprocess_image(sample.image_path, image_size=model_info["image_size"])
            image_tensor = image_tensor.to(device)

            with torch.enable_grad():
                cam = compute_gradcam(model_info["model"], image_tensor, args.threshold)
            with torch.no_grad():
                logits = model_info["model"](image_tensor)

            prediction = postprocess_prediction(
                logits,
                output_size=(original_image.shape[1], original_image.shape[0]),
                threshold=args.threshold,
            )
            gradcam_overlay = create_gradcam_overlay(original_image, cam)
            rows.append(
                {
                    "name": model_info["display_name"],
                    "prediction": prediction,
                    "gradcam": gradcam_overlay,
                }
            )

        grid = create_grid(sample.sample_id, original_image, ground_truth, rows)
        save_path = output_dir / f"{index:02d}_{sample.sample_id}_all_models_gradcam_grid.png"
        cv2.imwrite(str(save_path), grid)
        print(f"Saved {save_path}")

    print(f"All model comparison grids saved to {output_dir}")


if __name__ == "__main__":
    main()
