import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from deep_learning.registry import MODEL_REGISTRY          # remove if not needed


# ─────────────────────────────────────────────────────────
# Utility: swap the RGB stem for an arbitrary channel count
# ─────────────────────────────────────────────────────────
def _adapt_stem_conv(model: nn.Module, in_channels: int, pretrained: bool):
    """Replace EfficientNet’s first Conv so it accepts `in_channels`."""
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
                # Grey → average R-G-B kernels
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            else:
                # Anything else → fresh Kaiming init
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out",
                                        nonlinearity="relu")

    model.features[0][0] = new_conv
    return model


# ─────────────────────────────────────────────────────────
# Custom EfficientNet-B0 wrapper
# ─────────────────────────────────────────────────────────
class EfficientNetB0Custom(nn.Module):
    """
    EfficientNet-B0 that
      • accepts any channel count
      • resizes images to `target_size` inside `forward`
      • swaps the classifier for `num_classes`
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        target_size: int  = 224,   # canonical B0 size
        pretrained: bool = True,
        resize_mode: str = "bilinear",
    ):
        super().__init__()

        # Normalise target size
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.resize_mode = resize_mode

        # Backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # Stem adaptation if needed
        if in_channels != 3:
            self.backbone = _adapt_stem_conv(self.backbone, in_channels, pretrained)

        # New classifier
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_feats, num_classes)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Resize on-the-fly if the incoming size differs
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode=self.resize_mode,
                              align_corners=False)

        # 2. Standard EfficientNet flow
        return self.backbone(x)


# ─────────────────────────────────────────────────────────
# Factory function – mirrors your registry pattern
# ─────────────────────────────────────────────────────────
@MODEL_REGISTRY.register()
def EfficientNetB0(
    num_classes: int = 2,
    in_channels: int = 1,
    target_size: int  = 224,
    pretrained: bool = True,
    resize_mode: str = "bilinear",
):
    print("EfficientNetB0 is implemented to reshape whatever input size to the arbument target_size. Default 224")

    return EfficientNetB0Custom(
        num_classes=num_classes,
        in_channels=in_channels,
        target_size=target_size,
        pretrained=pretrained,
        resize_mode=resize_mode,
    )


# ─────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EfficientNetB0(num_classes=2, in_channels=1, target_size=224)

    # Dummy batch: 8 greyscale 100×100 images
    x = torch.randn(8, 1, 100, 100)
    logits = model(x)
    print(model)
    print("Output shape:", logits.shape)   # torch.Size([8, 2])
