# import torch
# import torch.nn as nn
# import spconv.pytorch as spconv
# import torch._dynamo
# from deep_learning.registry import MODEL_REGISTRY

# # Fall back to eager whenever Dynamo hits an spconv op it can't compile
# torch._dynamo.config.suppress_errors = True


# def dense_to_sparse(x: torch.Tensor) -> spconv.SparseConvTensor:
#     B, C, H, W = x.shape
#     mask = (x.abs().sum(dim=1) > 0)             # (B, H, W)
#     b_idxs, ys, xs = mask.nonzero(as_tuple=True)
#     coords = torch.stack([b_idxs, ys, xs], 1).int()  # (N_active, 3)
#     feats  = x[b_idxs, :, ys, xs]                   # (N_active, C)
#     return spconv.SparseConvTensor(
#         feats, coords,
#         spatial_shape=[H, W],
#         batch_size=B
#     )


# class SparseDownsample(nn.Module):
#     """1×1 strided SparseConv2d + BN for the skip branch."""
#     def __init__(self, in_ch, out_ch, stride):
#         super().__init__()
#         self.conv = spconv.SparseConv2d(in_ch, out_ch,
#                                         kernel_size=1,
#                                         stride=stride,
#                                         bias=False)
#         self.bn   = nn.BatchNorm1d(out_ch)

#     def forward(self, x):
#         x = self.conv(x)
#         feats = self.bn(x.features)
#         return x.replace_feature(feats)


# class SparseBottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_ch, out_ch, stride=1, downsample=None):
#         super().__init__()
#         # 1×1 SubMConv (preserves pattern)
#         self.conv1 = spconv.SubMConv2d(in_ch, out_ch,
#                                        kernel_size=1,
#                                        bias=False)
#         self.bn1   = nn.BatchNorm1d(out_ch)

#         # 3×3 — if stride==1 keep subm; else use SparseConv2d to downsample pattern
#         if stride == 1:
#             self.conv2 = spconv.SubMConv2d(out_ch, out_ch,
#                                            kernel_size=3,
#                                            padding=1,
#                                            bias=False)
#         else:
#             self.conv2 = spconv.SparseConv2d(out_ch, out_ch,
#                                              kernel_size=3,
#                                              stride=stride,
#                                              padding=1,
#                                              bias=False)
#         self.bn2 = nn.BatchNorm1d(out_ch)

#         # 1×1 SubMConv to expand channels
#         self.conv3 = spconv.SubMConv2d(out_ch,
#                                        out_ch * self.expansion,
#                                        kernel_size=1,
#                                        bias=False)
#         self.bn3 = nn.BatchNorm1d(out_ch * self.expansion)

#         self.relu       = nn.ReLU(inplace=True)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x

#         # -- conv1 --
#         out = self.conv1(x)
#         feats = self.relu(self.bn1(out.features))
#         out = out.replace_feature(feats)

#         # -- conv2 --
#         out = self.conv2(out)
#         feats = self.relu(self.bn2(out.features))
#         out = out.replace_feature(feats)

#         # -- conv3 --
#         out = self.conv3(out)
#         feats = self.bn3(out.features)
#         out = out.replace_feature(feats)

#         # -- skip branch --
#         if self.downsample is not None:
#             identity = self.downsample(x)

#         # -- add + final ReLU --
#         out = out + identity
#         feats = self.relu(out.features)
#         return out.replace_feature(feats)


# class SparseResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=2):
#         super().__init__()
#         self.in_ch = 64

#         # initial conv (pattern‐changing), BN, ReLU, pool
#         self.conv1 = spconv.SparseConv2d(1, 64,
#                                          kernel_size=7,
#                                          stride=2,
#                                          padding=3,
#                                          bias=False)
#         self.bn1   = nn.BatchNorm1d(64)
#         self.relu  = nn.ReLU(inplace=True)
#         self.pool  = spconv.SparseConv2d(64, 64,
#                                          kernel_size=3,
#                                          stride=2,
#                                          padding=1,
#                                          bias=False)

#         # residual layers
#         self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         # final dense classifier
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc      = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, out_ch, blocks, stride):
#         downsample = None
#         if stride != 1 or self.in_ch != out_ch * block.expansion:
#             downsample = SparseDownsample(self.in_ch,
#                                           out_ch * block.expansion,
#                                           stride)
#         layers = [block(self.in_ch, out_ch, stride, downsample)]
#         self.in_ch = out_ch * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_ch, out_ch))
#         return nn.Sequential(*layers)

#     @torch._dynamo.disable
#     def forward(self, x: torch.Tensor):
#         # 1) mask → sparse
#         x = dense_to_sparse(x)

#         # 2) initial conv + BN + ReLU
#         x = self.conv1(x)
#         feats = self.relu(self.bn1(x.features))
#         x = x.replace_feature(feats)

#         # 3) pool (no BN/ReLU)
#         x = self.pool(x)

#         # 4) residual stages
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         # 5) to dense → avgpool → fc
#         dense  = x.dense()
#         pooled = self.avgpool(dense)
#         flat   = pooled.view(pooled.size(0), -1)
#         return self.fc(flat)


# @MODEL_REGISTRY.register()
# def SparseResNet50(num_classes=2):
#     return SparseResNet(SparseBottleneck, [3, 4, 6, 3], num_classes=num_classes)


# if __name__ == "__main__":
#     device = torch.device("cuda")
#     model  = ResNet50_Sparse(num_classes=2).to(device)
#     x      = torch.randn(8, 1, 224, 224, device=device)
#     x[x < 0.1] = 0.0
#     print(model(x).shape)  # -> torch.Size([8, 2])


import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch._dynamo
from deep_learning.registry import MODEL_REGISTRY

# Fall back to eager whenever Dynamo hits an spconv op it can't compile
torch._dynamo.config.suppress_errors = True


class PixelThreshold(nn.Module):
    """Zero out all pixels in the dense input below `threshold`."""
    def __init__(self, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # make a copy so we don't destroy the original input
        x = x.clone()
        x[x < self.threshold] = 0.0
        return x


def dense_to_sparse(x: torch.Tensor) -> spconv.SparseConvTensor:
    B, C, H, W = x.shape
    mask = (x.abs().sum(dim=1) > 0)             # (B, H, W)
    b_idxs, ys, xs = mask.nonzero(as_tuple=True)
    coords = torch.stack([b_idxs, ys, xs], 1).int()  # (N_active, 3)
    feats  = x[b_idxs, :, ys, xs]                   # (N_active, C)
    return spconv.SparseConvTensor(
        feats, coords,
        spatial_shape=[H, W],
        batch_size=B
    )


class SparseDownsample(nn.Module):
    """1×1 strided SparseConv2d + BN for the skip branch,
       but use 3×3, padding=1 whenever stride>1 to match main‐branch pattern."""
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        if stride == 1:
            kernel_size, padding = 1, 0
        else:
            # match the main branch's conv2 (kernel=3, padding=1) when stride>1
            kernel_size, padding = 3, 1

        self.conv = spconv.SparseConv2d(in_ch, out_ch,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        feats = self.bn(x.features)
        return x.replace_feature(feats)


class SparseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        # 1×1 SubMConv (preserves pattern)
        self.conv1 = spconv.SubMConv2d(in_ch, out_ch,
                                       kernel_size=1,
                                       bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)

        # 3×3 — if stride==1 keep subm; else use SparseConv2d to downsample pattern
        if stride == 1:
            self.conv2 = spconv.SubMConv2d(out_ch, out_ch,
                                           kernel_size=3,
                                           padding=1,
                                           bias=False)
        else:
            self.conv2 = spconv.SparseConv2d(out_ch, out_ch,
                                             kernel_size=3,
                                             stride=stride,
                                             padding=1,
                                             bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # 1×1 SubMConv to expand channels
        self.conv3 = spconv.SubMConv2d(out_ch,
                                       out_ch * self.expansion,
                                       kernel_size=1,
                                       bias=False)
        self.bn3 = nn.BatchNorm1d(out_ch * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # -- conv1 --
        out = self.conv1(x)
        feats = self.relu(self.bn1(out.features))
        out = out.replace_feature(feats)

        # -- conv2 --
        out = self.conv2(out)
        feats = self.relu(self.bn2(out.features))
        out = out.replace_feature(feats)

        # -- conv3 --
        out = self.conv3(out)
        feats = self.bn3(out.features)
        out = out.replace_feature(feats)

        # -- skip branch --
        if self.downsample is not None:
            identity = self.downsample(x)

        # -- add + final ReLU --
        out = out + identity
        feats = self.relu(out.features)
        return out.replace_feature(feats)


class SparseResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes: int = 2,
        input_module: nn.Module = None,      # ← new
    ):
        super().__init__()
        # default threshold filter if none provided
        self.input_module = input_module or PixelThreshold(0.1)

        self.in_ch = 64

        # initial conv (pattern‐changing), BN, ReLU, pool
        self.conv1 = spconv.SparseConv2d(1, 64,
                                         kernel_size=7,
                                         stride=2,
                                         padding=3,
                                         bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = spconv.SparseConv2d(64, 64,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         bias=False)

        # residual layers
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # final dense classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch * block.expansion:
            downsample = SparseDownsample(self.in_ch,
                                          out_ch * block.expansion,
                                          stride)
        layers = [block(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    @torch._dynamo.disable
    def forward(self, x: torch.Tensor):
        # 0) pre‐process the dense input
        x = self.input_module(x)

        # 1) mask → sparse
        x = dense_to_sparse(x)

        # 2) initial conv + BN + ReLU
        x = self.conv1(x)
        feats = self.relu(self.bn1(x.features))
        x = x.replace_feature(feats)

        # 3) pool (no BN/ReLU)
        x = self.pool(x)

        # 4) residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 5) to dense → avgpool → fc
        dense  = x.dense()
        pooled = self.avgpool(dense)
        flat   = pooled.view(pooled.size(0), -1)
        return self.fc(flat)


@MODEL_REGISTRY.register()
def SparseResNet50(num_classes: int = 2, input_module: nn.Module = None):
    return SparseResNet(SparseBottleneck, [3, 4, 6, 3],
                        num_classes=num_classes,
                        input_module=input_module)


if __name__ == "__main__":
    device = torch.device("cuda")
    # default behavior: zero out all pixels < 0.1
    model  = SparseResNet50(num_classes=2).to(device)

    # or supply your own module, e.g. threshold 0.2
    # custom_filter = PixelThreshold(0.2)
    # model = SparseResNet50(num_classes=2, input_module=custom_filter).to(device)

    x      = torch.randn(8, 1, 224, 224, device=device)
    print("Before filter, min:", x.min().item())
    print("After  filter, min:", (model.input_module(x)).min().item())
    x[x < 0.1] = 0.0  # same effect as the default filter
    print(model(x).shape)  # -> torch.Size([8, 2])
