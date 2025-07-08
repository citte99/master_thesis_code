import torch

from .mass_component import MassComponent
import shared_utils.units as units

import shared_utils.units as units

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compute_dtype= torch.float32

# Store compute precision constants for reuse
const_compute_dtype = torch.float32
const_c = torch.tensor(units.c, device=device, dtype=compute_dtype)
const_pi4 = torch.tensor(4 * torch.pi, device=device, dtype=compute_dtype)
const_epsilon = torch.tensor(1e-8, device=device, dtype=compute_dtype)
const_zero = torch.tensor(0.0, device=device, dtype=compute_dtype)
        



class SIS(MassComponent):
    def __init__(self, param_tensor: torch.Tensor, precomputed_dict: dict, device="cuda", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.compute_dtype = torch.float32


        # Extract parameters 
        self.pos = torch.stack((param_tensor[:, 0], param_tensor[:, 1]), dim=-1)  # [B, 2]
        self.redshift = param_tensor[:, 2]  # [B]
        self.vel_disp = param_tensor[:, 3]  # [B]
        
        D_ls_index = precomputed_dict["param_map"].index("D_ls")
        D_s_index = precomputed_dict["param_map"].index("D_s")

        # Convert distances from Mpc -> kpc
        self.D_ls = precomputed_dict["params"][:, D_ls_index] * 1000  # [B]
        self.D_s = precomputed_dict["params"][:, D_s_index] * 1000


        self.c=const_c
        self.pi4=const_pi4
        self.epsilon=const_epsilon
        self.zero=const_zero
        
        
        
        # The precomputed einstein radius is supposed to be of the main mass.
        # We already have a problem in the pemd as we use the precomputed Theta E
        # But the pemd is not likely to be used as substucture. Still, need to fix that.


        vel_disp_c = self.vel_disp / self.c
        squared = vel_disp_c ** 2
        D_ratio = self.D_ls / self.D_s
        self.einstein_angle = self.pi4 * squared * D_ratio
        
        
        # Buffers will be allocated on first use
        self._initialized_buffers = False
        self._buffer_shapes = None
    
    def _initialize_buffers(self, batch_size, height, width):
        """Initialize all buffers at once with compute precision"""
        # Only initialize once for each shape configuration
        current_shapes = (batch_size, height, width)
        if self._initialized_buffers and self._buffer_shapes == current_shapes:
            return
            
        # Create all buffers in compute precision directly
        self._x_rel = torch.empty((batch_size, height, width, 2), 
                                  device=self.device, dtype=self.compute_dtype)
        self._r_squared = torch.empty((batch_size, height, width), 
                                      device=self.device, dtype=self.compute_dtype)
        self._r = torch.empty((batch_size, height, width, 1), 
                             device=self.device, dtype=self.compute_dtype)
        self._result = torch.empty((batch_size, height, width, 2), 
                                  device=self.device, dtype=self.compute_dtype)
        
        self._initialized_buffers = True
        self._buffer_shapes = current_shapes

   
    def deflection_angle(self, lens_grid, z_source=None):
        """Optimized implementation using pre-allocated buffers, minimizing copies,
        and supporting a single (H, W, 2) grid via natural broadcasting."""
        # 1) Ensure compute-dtype
        lens_grid_f32 = (
            lens_grid
            if lens_grid.dtype == self.compute_dtype
            else lens_grid.to(dtype=self.compute_dtype)
        )

        # 2) Determine batch_size, height, width from 3D or 4D input
        if lens_grid_f32.dim() == 3:
            # single grid broadcast to all halos
            B = self.pos.shape[0]
            H, W, _ = lens_grid_f32.shape
            batch_size, height, width = B, H, W
        elif lens_grid_f32.dim() == 4:
            batch_size, height, width, _ = lens_grid_f32.shape
            if batch_size != self.pos.shape[0]:
                raise ValueError(
                    f"Batch-size mismatch: got grid batch {batch_size}, "
                    f"but have {self.pos.shape[0]} halos"
                )
        else:
            raise ValueError(
                f"lens_grid must be 3D or 4D, got {lens_grid_f32.dim()}D"
            )

        # 3) (Re)initialize all buffers for this shape
        self._initialize_buffers(batch_size, height, width)

        # 4) Compute relative positions by broadcasting subtraction:
        #    lens_grid_f32: (H,W,2) or (B,H,W,2)
        #    pos_expanded:  (B,1,1,2)
        pos_expanded = self.pos.view(batch_size, 1, 1, 2)
        torch.sub(lens_grid_f32, pos_expanded, out=self._x_rel)  # -> (B,H,W,2)

        # 5) Squared radius r² = x² + y²
        x_rel = self._x_rel[..., 0]
        y_rel = self._x_rel[..., 1]
        torch.mul(x_rel, x_rel,         out=self._r_squared)   # x²
        torch.addcmul(self._r_squared, y_rel, y_rel, value=1.0, out=self._r_squared)  # + y²

        # 6) Radius r = sqrt(r²), with floor at epsilon
        #    self._r has shape (B, H, W, 1)
        #    we do sqrt into a squeezed view then restore the shape
        torch.sqrt(self._r_squared, out=self._r.view(batch_size, height, width))  
        torch.maximum(self._r.view(batch_size, height, width), self.epsilon,
                    out=self._r.view(batch_size, height, width))
        # _r now holds (B,H,W) in its squeezed storage; we reshape:
        self._r = self._r.view(batch_size, height, width, 1)

        # 7) Deflection: α = θ_E * (x_rel / r)
        einstein_angle_expanded = self.einstein_angle.view(batch_size, 1, 1, 1)
        torch.mul(einstein_angle_expanded, self._x_rel, out=self._result)
        torch.div(self._result, self._r,                out=self._result)

        # 8) Convert back to target precision if needed
        if self.dtype != self.compute_dtype:
            return self._result.to(dtype=self.dtype)
        return self._result
