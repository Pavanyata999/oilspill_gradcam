import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = 1, backbone: str = "resnet50", pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone_model = models.resnet50(weights=weights)
        elif backbone == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            backbone_model = models.resnet101(weights=weights)
        else:
            raise ValueError("Backbone must be 'resnet50' or 'resnet101'.")

        self.stem = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
        )
        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = backbone_model.layer4

        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.decoder = Decoder(aspp_channels=256, low_level_channels=256, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.stem(x)

        low_level_features = self.layer1(x)
        x = self.layer2(low_level_features)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = self.decoder(x, low_level_features)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.layer4[-1].conv3


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch1 = _ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.branch2 = _ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.branch3 = _ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.branch4 = _ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.project_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.project = _ConvBNReLU(out_channels * 5, out_channels, kernel_size=1, padding=0, dilation=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        ]

        pooled = self.pool(x)
        pooled = self.relu(self.project_pool(pooled))
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        features.append(pooled)

        return self.project(torch.cat(features, dim=1))


class Decoder(nn.Module):
    def __init__(self, aspp_channels: int, low_level_channels: int, num_classes: int) -> None:
        super().__init__()
        self.low_level_projection = _ConvBNReLU(low_level_channels, 48, kernel_size=1, padding=0, dilation=1)
        self.fuse = nn.Sequential(
            _ConvBNReLU(aspp_channels + 48, 256, kernel_size=3, padding=1, dilation=1),
            _ConvBNReLU(256, 256, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, low_level_features: torch.Tensor) -> torch.Tensor:
        low_level_features = self.low_level_projection(low_level_features)
        x = F.interpolate(x, size=low_level_features.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, low_level_features], dim=1)
        return self.fuse(x)


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
