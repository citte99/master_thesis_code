import torch
import torch.nn as nn

from .lens_model import LensModel
from .sources import GaussianBlob, SersicClumps
from shared_utils import recursive_to_tensor
import torch.nn.functional as F          # make sure this is imported


class LensingSystem(nn.Module):
    """
          
    """
    def __init__(self, config_dict, device):
        """
            The lensing system is a composition of a lens model and a source model.
            An example of the configuration dictionary is:

            config_dict = {
                "system_index": 0,
                "precomputed": {
                    "D_l": cosmo.angular_diameter_distance(lens_z).value,
                    "D_s": cosmo.angular_diameter_distance(source_z).value,
                    "D_ls": cosmo.angular_diameter_distance_z1z2(lens_z, source_z).value,
                },
                "lens_model": {
                    "num_substructures": 0,
                    "mass_components": [
                        {"type": "SIS",
                        "is_substructure": False,
                        "params": {"pos": torch.tensor([0.0, 0.0]), 
                                    "redshift": torch.tensor(lens_z), 
                                    "vel_disp": torch.tensor(220.)}
                        }
                    ]
                },
                "source_model": {
                    "type": "Gaussian_blob",
                    params: {
                        "I": 1.0,
                        "position_rad": torch.tensor([0.0, 0.0]),
                        "orient_rad": 0.0,
                        "q": 0.8,
                        "std_kpc": 0.1,
                        "redshift": source_z
                    }
                }
            }
        """
        super().__init__()

        # Move all the config dict to the device
        self.device = device
        self.config_dict = config_dict=recursive_to_tensor(config_dict, device)
        self.lens_model = LensModel(config_dict["lens_model"], precomp_dict=config_dict["precomputed"], device=device)
        source_mapping = {
            "Gaussian_blob": GaussianBlob,
            "Sersic_clumps": SersicClumps,
        }
        self.source_model = source_mapping[config_dict["source_model"]["type"]](config_dict["source_model"]["params"], precomp_dict=config_dict["precomputed"], device=device)
        self.source_redshift = config_dict["source_model"]["params"]["redshift"]



    
        
    def forward(self, lens_grid):
        """
           This forward calls the forward of the lens model and the source model
           and gives back the pixel values in the image plane.
        """
        source_grid = self.lens_model(lens_grid, self.source_redshift)
        #scatter plot for debugging the source grid tensor
        image_tensor = self.source_model(source_grid)
        return image_tensor
    

#     def forward(self, lens_grid, up=10):
#         """
#         lens_grid : [H, W, 2]  *or*  [B, H, W, 2]
#         returns   :             [H, W]      or  [B, H, W]
#         """
#         single = False
#         if lens_grid.ndim == 3:          # add fake batch dim
#             lens_grid = lens_grid.unsqueeze(0)
#             single = True                # remember to squeeze later

#         B, H, W, _ = lens_grid.shape
#         # ---------- build high-res grid (same code as before) ----------
#         x_min, x_max = lens_grid[..., 0].min(), lens_grid[..., 0].max()
#         y_min, y_max = lens_grid[..., 1].min(), lens_grid[..., 1].max()

#         xs = torch.linspace(x_min, x_max, W*up, device=lens_grid.device, dtype=lens_grid.dtype)
#         ys = torch.linspace(y_min, y_max, H*up, device=lens_grid.device, dtype=lens_grid.dtype)
#         Y_hr, X_hr = torch.meshgrid(ys, xs, indexing='ij')
#         grid_hr = torch.stack([X_hr, Y_hr], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

#         src_hr = self.lens_model(grid_hr, self.source_redshift)
#         img_hr = self.source_model(src_hr)            # [B, H*up, W*up]  or  [B,1,H*up,W*up]
#         if img_hr.ndim == 3:
#             img_hr = img_hr.unsqueeze(1)

#         img_lr = F.avg_pool2d(img_hr, kernel_size=up, stride=up).squeeze(1)  # [B, H, W]
#         return img_lr[0] if single else img_lr
