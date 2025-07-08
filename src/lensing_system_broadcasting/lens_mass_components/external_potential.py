import torch

from .mass_component import MassComponent
from shared_utils import _arcsec_to_rad




class ExternalPotential(MassComponent):
    def __init__(self, param_tensor: torch.Tensor, precomputed_dict: dict, device="cuda", dtype=torch.float32):
        """
        For the ExternalPotential, the input tensor is expected to be of shape [B, 4]:
            [[shear_center_x, shear_center_y, shear_strength, shear_angle_arcsec],
             [shear_center_x, shear_center_y, shear_strength, shear_angle_arcsec],
             ...]
        """
        super().__init__(device, dtype=dtype)
        
    
        
        # Extract batched parameters.
        # shear_center: shape [B, 2]
        self.shear_center = torch.stack((param_tensor[:, 0], param_tensor[:, 1]), dim=-1)
        # shear_strength: shape [B]
        self.ss = param_tensor[:, 2]
        # Convert shear angle from arcseconds to radians.
        self.sa = _arcsec_to_rad(param_tensor[:, 3])
        self.compute_dtype = torch.float32
        # Pre-compute doubled angles and their trig functions
        self.sa2 = 2 * self.sa
        self.cos2sa = -torch.cos(self.sa2)
        self.sin2sa = -torch.sin(self.sa2)
        
        # Pre-compute constants for tensor operations
        self.two = torch.tensor(2.0, device=device, dtype=self.compute_dtype)
        self.minus_one = torch.tensor(-1.0, device=device, dtype=self.compute_dtype)
    
    def deflection_angle(self, lens_grid, precomp_dict=None, z_source=None):
        """
        Compute the batched deflection angle for the external shear.
        
        Parameters:
          lens_grid: Tensor of shape [B, H, W, 2] representing the grid in the lens plane.
          precomp_dict: Dummy argument for uniformity with other mass components.
          z_source: Dummy argument for clarity.
          
        Returns:
          A tensor of deflection angles with shape [B, H, W, 2].
        """
        # Upcast input grid if needed
        lens_grid_compute = lens_grid if lens_grid.dtype == self.compute_dtype else lens_grid.to(dtype=self.compute_dtype)
        
        # Reshape shear_center to [B, 1, 1, 2] for broadcasting.
        shear_center = self.shear_center.view(-1, 1, 1, 2)
        
        # Compute relative positions.
        xrel = lens_grid_compute - shear_center
        
        # Reshape shear strength and pre-computed trig values for broadcasting.
        ss = self.ss.view(-1, 1, 1)
        cs2 = self.cos2sa.view(-1, 1, 1)
        sn2 = self.sin2sa.view(-1, 1, 1)
        
        # Extract x and y components for clarity
        x = xrel[..., 0]
        y = xrel[..., 1]
        
        # Compute the deflection angle components more efficiently
        alpha_x = ss * (cs2 * x + sn2 * y)
        alpha_y = ss * (sn2 * x - cs2 * y)
        
        # Stack the components to form the vector field.
        alpha = torch.stack((alpha_x, alpha_y), dim=-1)
        
        # Convert to target precision only at the end if needed
        if self.dtype != self.compute_dtype:
            result = alpha.to(dtype=self.dtype)
        else:
            result = alpha
            
        return result