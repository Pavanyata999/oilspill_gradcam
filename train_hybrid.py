import argparse
import os
import time
from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import DEFAULT_IMAGE_SIZE, create_data_loaders
from models.deeplabv3_model import DeepLabV3Plus
from models.segnet_model import SegNet
from utils import (
    CombinedBCEDiceLoss,
    calculate_metrics,
    count_parameters,
    create_output_directory,
    get_device,
    save_model_checkpoint,
)


def create_model(model_type: str = "deeplabv3_plus", backbone: str = "resnet50", pretrained: bool = True):
    if model_type == "deeplabv3_plus":
        return DeepLabV3Plus(num_classes=1, backbone=backbone, pretrained=pretrained)
    if model_type == "segnet":
        return SegNet(num_classes=1, in_channels=3)
    if model_type == "hybrid":
        from models.hybrid_models import HybridOilSpillModel as HybridModel

        return HybridModel(num_classes=1)
    raise ValueError(f"Unknown model type: {model_type}")


def run_epoch(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, Dict[str, float]]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    progress = tqdm(data_loader, desc="Train" if is_training else "Val", leave=False)

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            metrics = calculate_metrics(logits.detach(), masks)
            total_loss += loss.item()
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{metrics['dice']:.4f}",
                iou=f"{metrics['iou']:.4f}",
            )

    num_batches = len(data_loader)
    return (
        total_loss / max(num_batches, 1),
        {
            "dice": total_dice / max(num_batches, 1),
            "iou": total_iou / max(num_batches, 1),
        },
    )


def train_model(args) -> str:
    device = get_device()
    output_dir = create_output_directory(args.output_dir, args.model_type)

    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        satellite_type=args.satellite_type,
    )

    model = create_model(
        model_type=args.model_type,
        backbone=args.backbone,
        pretrained=args.pretrained,
    ).to(device)

    criterion = CombinedBCEDiceLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "checkpoints", "best_model.pth")

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=None,
        )
        scheduler.step(val_loss)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("dice/train", train_metrics["dice"], epoch)
        writer.add_scalar("dice/val", val_metrics["dice"], epoch)
        writer.add_scalar("iou/train", train_metrics["iou"], epoch)
        writer.add_scalar("iou/val", val_metrics["iou"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_dice={train_metrics['dice']:.4f} val_dice={val_metrics['dice']:.4f} | "
            f"train_iou={train_metrics['iou']:.4f} val_iou={val_metrics['iou']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        metrics_payload = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_metrics["dice"],
            "val_dice": val_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_iou": val_metrics["iou"],
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics_payload,
                save_path=best_checkpoint_path,
                model_type=args.model_type,
                image_size=args.image_size,
            )
            print(f"Saved best checkpoint to {best_checkpoint_path}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            periodic_path = os.path.join(output_dir, "checkpoints", f"epoch_{epoch:03d}.pth")
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics_payload,
                save_path=periodic_path,
                model_type=args.model_type,
                image_size=args.image_size,
            )

    final_checkpoint_path = os.path.join(output_dir, "checkpoints", "last_model.pth")
    save_model_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        metrics=metrics_payload,
        save_path=final_checkpoint_path,
        model_type=args.model_type,
        image_size=args.image_size,
    )
    writer.close()
    print(f"Training finished. Best checkpoint: {best_checkpoint_path}")
    return best_checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAR oil spill segmentation models")
    parser.add_argument("--data_dir", type=str, default="dataset/SOS_dataset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_type", type=str, default="hybrid", choices=["hybrid", "deeplabv3_plus", "segnet"])
    parser.add_argument("--satellite_type", type=str, default="both", choices=["both", "palsar", "sentinel"])
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
