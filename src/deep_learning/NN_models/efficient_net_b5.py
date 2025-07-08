import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from deep_learning.registry import MODEL_REGISTRY   # remove if you don’t use a registry


# ────────────────────────────────────────────────────────────────────────────────
# Helper: adapt the RGB stem to an arbitrary channel count
# ────────────────────────────────────────────────────────────────────────────────
def _adapt_stem_conv(model: nn.Module, in_channels: int, pretrained: bool):
    """
    Replace EfficientNet's first conv layer so the model can digest
    `in_channels` instead of the default 3.
    """
    old_conv: nn.Conv2d = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    if pretrained:
        with torch.no_grad():
            if in_channels == 1:
                # Grey input → average RGB kernels
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            else:
                # Any other channel count → fresh Kaiming init
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

    model.features[0][0] = new_conv
    return model


# ────────────────────────────────────────────────────────────────────────────────
# Main wrapper: EfficientNet-B5 with flexible channels / classes / input size
# ────────────────────────────────────────────────────────────────────────────────
class EfficientNetB5Custom(nn.Module):
    """
    EfficientNet-B5 that:
      • handles any number of input channels
      • swaps the classifier for `num_classes`
      • forcibly resizes every image to `target_size` inside `forward`
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        target_size: int | tuple[int, int] = 456,
        pretrained: bool = True,
        resize_mode: str = "bilinear",
    ):
        super().__init__()

        # Accept both int and (H, W) tuples
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.resize_mode = resize_mode

        # Backbone
        weights = EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b5(weights=weights)

        # Adapt the stem if needed
        if in_channels != 3:
            self.backbone = _adapt_stem_conv(self.backbone, in_channels, pretrained)

        # Replace classification head
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_feats, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1️⃣  Upsample / downsample to the canonical resolution
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(
                x, size=self.target_size, mode=self.resize_mode, align_corners=False
            )

        # 2️⃣  Standard EfficientNet forward
        return self.backbone(x)


# ────────────────────────────────────────────────────────────────────────────────
# Factory function (keeps parity with your MODEL_REGISTRY pattern)
# ────────────────────────────────────────────────────────────────────────────────
@MODEL_REGISTRY.register()
def EfficientNetB5(
    num_classes: int = 2,
    in_channels: int = 1,
    target_size: int | tuple[int, int] = 456,
    pretrained: bool = True,
    resize_mode: str = "bilinear",
):
    print("EfficientNetB5 is implemented to reshape whatever input size to the arbument target_size. Default 456")
    return EfficientNetB5Custom(
        num_classes=num_classes,
        in_channels=in_channels,
        target_size=target_size,
        pretrained=pretrained,
        resize_mode=resize_mode,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Example usage ─ run this file directly to test
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create the model: 1-channel input, 2 classes, force 256×256 inside
    model = EfficientNetB5(num_classes=2, in_channels=1, target_size=256)

    # Dummy batch: 8 grayscale images, raw size 100×100
    x = torch.randn(8, 1, 100, 100)
    logits = model(x)

    print(model)
    print("Output shape:", logits.shape)   # → torch.Size([8, 2])
