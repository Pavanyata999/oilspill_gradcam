import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_IMAGE_SIZE = 256
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageMaskPair:
    image_path: str
    mask_path: Optional[str]
    satellite_type: str
    sample_id: str


def get_transforms(mode: str = "train", image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    transforms = [A.Resize(image_size, image_size)]

    if mode == "train":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

    transforms.extend(
        [
            A.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def _sample_id_from_name(file_name: str) -> str:
    stem = Path(file_name).stem
    if stem.endswith("_sat"):
        return stem[:-4]
    if stem.endswith("_mask"):
        return stem[:-5]
    return stem


def _iter_image_files(directory: Path) -> Sequence[Path]:
    return sorted(
        [
            file_path
            for file_path in directory.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS and file_path.stem.endswith("_sat")
        ]
    )


def collect_image_mask_pairs(
    split_dir: str,
    satellite_type: str = "both",
    require_masks: bool = True,
) -> List[ImageMaskPair]:
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

    satellites = []
    if satellite_type in {"both", "palsar"}:
        satellites.append("palsar")
    if satellite_type in {"both", "sentinel"}:
        satellites.append("sentinel")

    pairs: List[ImageMaskPair] = []
    skipped = 0

    for satellite in satellites:
        satellite_dir = split_path / satellite
        if not satellite_dir.exists():
            warnings.warn(f"Satellite directory missing and skipped: {satellite_dir}")
            continue

        sat_dir = satellite_dir / "sat"
        gt_dir = satellite_dir / "gt"
        if sat_dir.exists():
            image_dir = sat_dir
            mask_dir = gt_dir if gt_dir.exists() else None
        else:
            image_dir = satellite_dir
            mask_dir = satellite_dir

        for image_path in _iter_image_files(image_dir):
            sample_id = _sample_id_from_name(image_path.name)
            mask_path = None

            if mask_dir is not None:
                candidate = mask_dir / f"{sample_id}_mask.png"
                if candidate.exists():
                    mask_path = str(candidate)

            if require_masks and mask_path is None:
                skipped += 1
                warnings.warn(f"Mask missing for image {image_path}; skipping sample.")
                continue

            pairs.append(
                ImageMaskPair(
                    image_path=str(image_path),
                    mask_path=mask_path,
                    satellite_type=satellite,
                    sample_id=sample_id,
                )
            )

    if not pairs:
        raise RuntimeError(f"No valid image/mask pairs found in {split_dir}")

    if skipped:
        warnings.warn(f"Skipped {skipped} samples with missing masks in {split_dir}")

    return pairs


class OilSpillDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        satellite_type: str = "both",
        transform: Optional[A.Compose] = None,
        mode: str = "train",
        require_masks: bool = True,
        return_metadata: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.satellite_type = satellite_type
        self.transform = transform or get_transforms(mode=mode, image_size=DEFAULT_IMAGE_SIZE)
        self.mode = mode
        self.require_masks = require_masks
        self.return_metadata = return_metadata
        self.samples = collect_image_mask_pairs(
            split_dir=data_dir,
            satellite_type=satellite_type,
            require_masks=require_masks,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_valid_sample(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], ImageMaskPair]:
        last_error = None

        for offset in range(len(self.samples)):
            sample = self.samples[(idx + offset) % len(self.samples)]
            image = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
            if image is None:
                last_error = f"Failed to read image: {sample.image_path}"
                warnings.warn(last_error)
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = None
            if sample.mask_path is not None:
                mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    last_error = f"Failed to read mask: {sample.mask_path}"
                    warnings.warn(last_error)
                    continue
                mask = (mask > 0).astype(np.float32)

            return image, mask, sample

        raise RuntimeError(last_error or "No readable samples found in dataset.")

    def __getitem__(self, idx: int):
        image, mask, sample = self._load_valid_sample(idx)

        if mask is None:
            transformed = self.transform(image=image)
            image_tensor = transformed["image"].float()
            metadata = {
                "image_path": sample.image_path,
                "mask_path": sample.mask_path,
                "sample_id": sample.sample_id,
                "satellite_type": sample.satellite_type,
            }
            if self.return_metadata:
                return image_tensor, metadata
            return image_tensor

        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"].float()
        mask_tensor = transformed["mask"]

        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.from_numpy(mask_tensor)

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        mask_tensor = (mask_tensor > 0.5).float()
        metadata = {
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
            "sample_id": sample.sample_id,
            "satellite_type": sample.satellite_type,
        }

        if self.return_metadata:
            return image_tensor, mask_tensor, metadata
        return image_tensor, mask_tensor


def create_data_loaders(
    data_dir: str,
    batch_size: int = 4,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = 0,
    satellite_type: str = "both",
) -> Tuple[DataLoader, DataLoader]:
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "test")

    train_dataset = OilSpillDataset(
        data_dir=train_dir,
        satellite_type=satellite_type,
        transform=get_transforms("train", image_size=image_size),
        mode="train",
        require_masks=True,
    )
    val_dataset = OilSpillDataset(
        data_dir=val_dir,
        satellite_type=satellite_type,
        transform=get_transforms("val", image_size=image_size),
        mode="val",
        require_masks=True,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
