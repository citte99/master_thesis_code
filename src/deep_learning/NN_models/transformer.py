import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.utils.data import Dataset, DataLoader
from catalog_manager import CatalogManager
from lensing_system import LensingSystem
from shared_utils import recursive_to_tensor, _grid_lens
import torch.optim as optim
from astropy.io import fits
import math
import gc
from deep_learning.registry import MODEL_REGISTRY



class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding - Splits image into patches and projects them to embedding space
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Handle non-standard input sizes
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            # Calculate padding needed
            pad_h = self.patch_size[0] - (H % self.patch_size[0])
            pad_w = self.patch_size[1] - (W % self.patch_size[1])
            
            # Only pad if needed
            if pad_h < self.patch_size[0] and pad_w < self.patch_size[1]:
                x = F.pad(x, (0, pad_w, 0, pad_h))
                # Update dimensions
                _, _, H, W = x.shape
        
        # Project patches
        x = self.proj(x)  # B, E, H', W'
        
        # Reshape to sequence format
        x = x.flatten(2).transpose(1, 2)  # B, N, E
        
        return x


class Attention(nn.Module):
    """
    Multi-head Self-Attention with optimized memory usage
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Combined query, key, value projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Compute query, key, value for all heads in batch
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        # Compute attention 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Project back to embedding dimension
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    MLP block for transformer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer encoder block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        # First normalization and attention
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        # Second normalization and MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )
        
    def forward(self, x):
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x

@MODEL_REGISTRY.register()
class VisionTransformer(nn.Module):
    """
    Vision Transformer for high-resolution interferometric images
    
    This implementation includes:
    - Flexible patching for arbitrary image sizes
    - Memory-efficient processing
    - Configurable architecture
    """
    def __init__(
        self, 
        img_size=224,
        patch_size=16, 
        in_channels=1, 
        num_classes=2, 
        embed_dim=768,
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True, 
        representation_size=None,
        distilled=False,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.0, 
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (2 if distilled else 1), embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        # Final normalization layer
        self.norm = norm_layer(embed_dim)
        
        # Classification head
        self.distilled = distilled
        self.num_classes = num_classes
        
        # Pre-logits if needed
        if representation_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()
        
        # Setup for either standard or distilled classification
        if distilled:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize class token and position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        self.apply(self._init_weights_linear)
    
    def _init_weights_linear(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        # Get image batch size
        B = x.shape[0]
        
        # Apply patch embedding
        x = self.patch_embed(x)
        
        # Interpolate position embeddings for non-standard sizes
        if x.size(1) != self.pos_embed.size(1) - 1:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            pos_embed = self.pos_embed[:, 1:, :]
            
            # Interpolate position embeddings to the actual sequence length
            seq_length = x.size(1)
            pos_embed_width = int(math.sqrt(pos_embed.size(1)))
            pos_embed = pos_embed.reshape(1, pos_embed_width, pos_embed_width, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, 
                size=int(math.sqrt(seq_length)), 
                mode='bicubic', 
                align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, x.size(2))
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)
        else:
            pos_embed = self.pos_embed
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        
        if self.distilled:
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Process through transformer blocks
        x = self.blocks(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Return appropriate tokens
        if self.distilled:
            return x[:, 0], x[:, 1]
        else:
            return x[:, 0]
    
    def forward(self, x):
        # Process through vision transformer
        if self.distilled:
            x, x_dist = self.forward_features(x)
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            return (x + x_dist) / 2
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x


@MODEL_REGISTRY.register()
def VisionTransformer_Custom(num_classes=2, img_size=224, patch_size=32, embed_dim=384, 
                             depth=6, num_heads=6, mlp_ratio=2.0, drop_rate=0.1, **kwargs):
    """
    Factory function to create a Vision Transformer with custom configuration.
    Follows similar pattern to ResNet50 factory function for consistency.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        img_size (int): Input image size (default: 224)
        patch_size (int): Size of each patch (default: 32)
        embed_dim (int): Embedding dimension (default: 384)
        depth (int): Number of transformer blocks (default: 6)
        num_heads (int): Number of attention heads (default: 6)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim (default: 2.0)
        drop_rate (float): Dropout rate (default: 0.1)
        **kwargs: Additional arguments to pass to VisionTransformer
        
    Returns:
        VisionTransformer: Configured model instance
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=1,  # Assuming grayscale images as in ResNet
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop_rate=drop_rate,
        **kwargs
    )

# Example usage:
if __name__ == "__main__":
    model = VisionTransformer_Custom(num_classes=2)
    print(model)
    
    # Create a dummy input: batch of 8 images, 1 channel, size 224x224
    x = torch.randn(8, 1, 224, 224)
    logits = model(x)
    print("Output shape:", logits.shape)