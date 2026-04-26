import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2

DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.view(probabilities.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def _flatten_binary_scores(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, ...]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1).float()

    intersection = (predictions * targets).sum(dim=1)
    pred_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum - intersection
    return predictions, targets, intersection, pred_sum, target_sum, union


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> float:
    _, _, intersection, pred_sum, target_sum, _ = _flatten_binary_scores(logits, targets, threshold=threshold)
    score = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return score.mean().item()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> float:
    _, _, intersection, _, _, union = _flatten_binary_scores(logits, targets, threshold=threshold)
    score = (intersection + eps) / (union + eps)
    return score.mean().item()


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    return {
        "dice": dice_score(logits, targets, threshold=threshold),
        "iou": iou_score(logits, targets, threshold=threshold),
    }


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def create_output_directory(base_dir: str = "outputs", model_name: str = "model") -> str:
    output_dir = ensure_dir(os.path.join(base_dir, model_name))
    ensure_dir(os.path.join(output_dir, "checkpoints"))
    ensure_dir(os.path.join(output_dir, "predictions"))
    ensure_dir(os.path.join(output_dir, "visualizations"))
    ensure_dir(os.path.join(output_dir, "logs"))
    return output_dir


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    model_type: str,
    image_size: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "model_type": model_type,
        "image_size": image_size,
    }
    ensure_dir(str(Path(save_path).parent))
    torch.save(checkpoint, save_path)


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * DEFAULT_STD + DEFAULT_MEAN
    return np.clip(image, 0.0, 1.0)


def preprocess_image(image_path: str, image_size: int = 256) -> Tuple[torch.Tensor, np.ndarray]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=DEFAULT_MEAN.tolist(), std=DEFAULT_STD.tolist()),
            ToTensorV2(),
        ]
    )
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).float()
    return image_tensor, original_image


def postprocess_prediction(logits: torch.Tensor, output_size: Tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    probabilities = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    mask = (probabilities >= threshold).astype(np.uint8) * 255
    width, height = output_size
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def save_mask(mask: np.ndarray, output_path: str) -> None:
    ensure_dir(str(Path(output_path).parent))
    cv2.imwrite(output_path, mask)


def save_visualization(
    original_image: np.ndarray,
    prediction_mask: np.ndarray,
    save_path: str,
    ground_truth_mask: Optional[np.ndarray] = None,
    gradcam_overlay: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> None:
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    panels = [("Original", _to_bgr(original_image))]
    if ground_truth_mask is not None:
        panels.append(("Ground Truth", _to_bgr(ground_truth_mask)))
    panels.append(("Prediction", _to_bgr(prediction_mask)))
    if gradcam_overlay is not None:
        panels.append(("Grad-CAM", _to_bgr(gradcam_overlay)))

    height = max(panel.shape[0] for _, panel in panels)
    width = max(panel.shape[1] for _, panel in panels)
    label_height = 40
    spacer = 10
    rendered = []

    for panel_title, panel_image in panels:
        resized = cv2.resize(panel_image, (width, height), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((height + label_height, width, 3), 255, dtype=np.uint8)
        canvas[label_height:, :, :] = resized
        cv2.putText(canvas, panel_title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        rendered.append(canvas)

    combined = rendered[0]
    for panel in rendered[1:]:
        separator = np.full((combined.shape[0], spacer, 3), 255, dtype=np.uint8)
        combined = np.hstack([combined, separator, panel])

    if title:
        title_canvas = np.full((50, combined.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(title_canvas, title, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        combined = np.vstack([title_canvas, combined])

    ensure_dir(str(Path(save_path).parent))
    cv2.imwrite(save_path, combined)
