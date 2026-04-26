import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from data_loader import collect_image_mask_pairs
from train_hybrid import create_model
from utils import (
    create_output_directory,
    get_device,
    load_model_checkpoint,
    postprocess_prediction,
    preprocess_image,
    save_visualization,
)


class BinarySegmentationTarget:
    def __init__(self, mask: np.ndarray):
        self.mask = torch.from_numpy(mask.astype(np.float32))

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        output = torch.sigmoid(model_output[0, 0])
        mask = self.mask.to(model_output.device)
        return (output * mask).sum()


def _load_mask(mask_path: Optional[str], shape: tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return cv2.resize((mask > 0).astype(np.uint8) * 255, shape, interpolation=cv2.INTER_NEAREST)


def generate_gradcam(args) -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'grad-cam' package is required for Grad-CAM generation. "
            "Install it with: pip install grad-cam"
        ) from exc

    device = get_device()
    model = create_model(model_type=args.model_type, backbone=args.backbone, pretrained=False).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, device=device)
    image_size = checkpoint.get("image_size", args.image_size)
    model.eval()

    if not hasattr(model, "get_gradcam_target_layer") and args.model_type != 'hybrid':
        raise ValueError(f"Model type '{args.model_type}' does not expose a Grad-CAM target layer.")

    output_dir = create_output_directory(args.output_dir, args.model_type)
    visualization_dir = Path(output_dir) / "visualizations"

    target_layer = model.get_gradcam_target_layer()
    samples = collect_image_mask_pairs(args.data_dir, satellite_type=args.satellite_type, require_masks=False)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for sample in samples:
            image_tensor, original_image = preprocess_image(sample.image_path, image_size=image_size)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                logits = model(image_tensor)
            probabilities = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            target_mask = (probabilities >= args.threshold).astype(np.float32)

            if target_mask.sum() == 0:
                target_mask = probabilities.astype(np.float32)

            targets = [BinarySegmentationTarget(target_mask)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

            original_float = original_image.astype(np.float32) / 255.0
            cam_overlay = show_cam_on_image(original_float, grayscale_cam, use_rgb=True)
            prediction_mask = postprocess_prediction(
                logits,
                output_size=(original_image.shape[1], original_image.shape[0]),
                threshold=args.threshold,
            )
            gt_mask = _load_mask(sample.mask_path, (original_image.shape[1], original_image.shape[0]))

            save_visualization(
                original_image=original_image,
                ground_truth_mask=gt_mask,
                prediction_mask=prediction_mask,
                gradcam_overlay=cam_overlay,
                save_path=str(visualization_dir / f"{sample.sample_id}_gradcam.png"),
                title=f"{sample.satellite_type} - {sample.sample_id}",
            )

    print(f"Grad-CAM visualizations saved to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for SAR oil spill segmentation")
    parser.add_argument("--data_dir", type=str, default="dataset/SOS_dataset/test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_type", type=str, default="deeplabv3_plus", choices=["deeplabv3_plus", "segnet", "hybrid"])
    parser.add_argument("--satellite_type", type=str, default="both", choices=["both", "palsar", "sentinel"])
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    generate_gradcam(build_parser().parse_args())
