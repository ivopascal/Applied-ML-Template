import torch.nn as nn
import torchvision.models as models
from torch import Tensor


class CNNBackbone(nn.Module):
    """
    A ResNet-34 backbone with an upsampling head for depth estimation.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """initialize the CNN backbone."""
        super().__init__()
        # Load ResNet-34 and remove classifier
        resnet = models.resnet34(pretrained=pretrained)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # Head
        self.head = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(
                scale_factor=4,
                mode="bilinear",
                align_corners=False
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the backbone and head.
        Ensures output matches input resolution.
        """
        features = self.backbone(x)
        output = self.head(features)

        # Resize to match input resolution if needed
        if output.shape[2:] != x.shape[2:]:
            output = nn.functional.interpolate(
                output,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        return output
