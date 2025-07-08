import torch

"""
This function was moved to shared utils.
"""
# def recursive_to_tensor(value, device):
#     # If value is a dictionary, process each key/value pair recursively.
#     if isinstance(value, dict):
#         return {k: recursive_to_tensor(v, device) for k, v in value.items()}
    
#     # If value is a list, process each element recursively.
#     elif isinstance(value, list):
#         return [recursive_to_tensor(item, device) for item in value]
    
#     # If value is a tuple, process each element recursively and return a tuple.
#     elif isinstance(value, tuple):
#         return tuple(recursive_to_tensor(item, device) for item in value)
    
#     # Try to convert numerical values (or lists/arrays of numbers) to a tensor.
#     else:
#         try:
#             # torch.as_tensor will convert if possible.
#             return torch.as_tensor(value, device=device)
#         except Exception:
#             # If conversion fails (e.g., for strings), return the original value.
#             return value
        




import random, torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class Crop80RotatePadBatch(torch.nn.Module):
    """
    • Randomly crop 0.8 × H and 0.8 × W
    • Rotate by ±`degrees` (bilinear, no expand)
    • Pad so the canvas is back to the original H × W
    Works on:
      • PIL images   → returns PIL
      • 3-D tensors  (C,H,W) → returns tensor
      • 4-D tensors  (B,C,H,W) → returns tensor batch
    """
    def __init__(self,
                 degrees=30,
                 crop_scale=0.8,
                 interpolation=InterpolationMode.BILINEAR,
                 fill=0):
        super().__init__()
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)
        self.degrees = degrees
        self.crop_scale = crop_scale
        self.interp = interpolation
        self.fill = fill

    # ---------- helpers ----------------------------------------------------
    def _transform_single(self, img):
        # --- get original size (works for PIL or tensor) -------------------
        if isinstance(img, torch.Tensor):
            _, orig_h, orig_w = img.shape
        else:                             # PIL.Image
            orig_w, orig_h = img.size

        # --- 1. random crop ------------------------------------------------
        tgt_w, tgt_h = int(orig_w * self.crop_scale), int(orig_h * self.crop_scale)
        left = random.randint(0, orig_w - tgt_w)
        top  = random.randint(0, orig_h - tgt_h)
        img  = F.crop(img, top, left, tgt_h, tgt_w)

        # --- 2. random rotation -------------------------------------------
        angle = random.uniform(*self.degrees)
        img   = F.rotate(img, angle,
                         expand=False,
                         interpolation=self.interp,
                         fill=self.fill)

        # --- 3. pad back to original size ---------------------------------
        pad_left   = (orig_w - img.shape[-1]) // 2
        pad_top    = (orig_h - img.shape[-2]) // 2
        pad_right  = orig_w - img.shape[-1] - pad_left
        pad_bottom = orig_h - img.shape[-2] - pad_top
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom),
                    fill=self.fill)

        return img

    # ---------- main entry point ------------------------------------------
    def forward(self, imgs):
        # (B,C,H,W) → list comprehension, then stack back
        if isinstance(imgs, torch.Tensor) and imgs.ndim == 4:
            return torch.stack([self._transform_single(img) for img in imgs])
        # single PIL image or (C,H,W) tensor
        return self._transform_single(imgs)


train_tf = Crop80RotatePadBatch(degrees=30)   # no ToTensor(); you already have tensors
