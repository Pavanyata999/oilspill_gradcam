import argparse
from typing import Optional

import cv2
import numpy as np
import torch

from data_loader import collect_image_mask_pairs
from models.deeplabv3_model import DeepLabV3Plus
from models.segnet_model import SegNet
from utils import (
    create_output_directory,
    get_device,
    load_model_checkpoint,
    postprocess_prediction,
    preprocess_image,
    save_mask,
    save_visualization,
)


def create_model(model_type: str = "deeplabv3_plus", backbone: str = "resnet50", pretrained: bool = False):
    if model_type == "deeplabv3_plus":
        return DeepLabV3Plus(num_classes=1, backbone=backbone, pretrained=pretrained)
    if model_type == "segnet":
        return SegNet(num_classes=1, in_channels=3)
    if model_type == "hybrid":
        from models.hybrid_models import HybridOilSpillModel

        return HybridOilSpillModel(num_classes=1)
    raise ValueError(f"Unknown model type: {model_type}")


def _load_mask(mask_path: Optional[str]) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return ((mask > 0).astype(np.uint8) * 255)


def run_inference(args) -> None:
    device = get_device()
    model = create_model(model_type=args.model_type, backbone=args.backbone, pretrained=False).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, device=device)
    image_size = checkpoint.get("image_size", args.image_size)
    model.eval()

    output_dir = create_output_directory(args.output_dir, args.model_type)
    prediction_dir = Path(output_dir) / "predictions"
    visualization_dir = Path(output_dir) / "visualizations"

    samples = collect_image_mask_pairs(args.data_dir, satellite_type=args.satellite_type, require_masks=False)
    if args.limit is not None:
        samples = samples[: args.limit]

    with torch.no_grad():
        for sample in samples:
            image_tensor, original_image = preprocess_image(sample.image_path, image_size=image_size)
            logits = model(image_tensor.to(device))
            prediction_mask = postprocess_prediction(
                logits,
                output_size=(original_image.shape[1], original_image.shape[0]),
                threshold=args.threshold,
            )

            save_mask(prediction_mask, str(prediction_dir / f"{sample.sample_id}_pred.png"))
            ground_truth = _load_mask(sample.mask_path)
            save_visualization(
                original_image=original_image,
                prediction_mask=prediction_mask,
                ground_truth_mask=ground_truth,
                save_path=str(visualization_dir / f"{sample.sample_id}_comparison.png"),
                title=f"{sample.satellite_type} - {sample.sample_id}",
            )

    print(f"Inference completed. Results saved to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run segmentation inference on SAR test images")
    parser.add_argument("--data_dir", type=str, default="dataset/SOS_dataset/test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_type", type=str, default="deeplabv3_plus", choices=["deeplabv3_plus", "segnet", "hybrid"])
    parser.add_argument("--satellite_type", type=str, default="both", choices=["both", "palsar", "sentinel"])
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    return parser


if __name__ == "__main__":
    run_inference(build_parser().parse_args())
