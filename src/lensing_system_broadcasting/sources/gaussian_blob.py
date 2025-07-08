import torch
from .source_model import SourceModel
from .util import _translate_rotate

class GaussianBlob(SourceModel):
    """
    Memory-efficient, batched Gaussian blob source model
    """
    def __init__(self, param_tensor: torch.Tensor, precomputed_dict: dict, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device

        # Parameter tensors (batched)
                 # shape [B]
        self.position_rad_tensor = param_tensor[:, 0:2]  # shape [B,2]
        self.I_tensor            = param_tensor[:, 2]    
        self.orient_rad_tensor   = param_tensor[:, 3]             # shape [B]
        self.q_tensor            = param_tensor[:, 4]             # shape [B]
        self.std_kpc_tensor      = param_tensor[:, 5]             # shape [B]
        #unused the following
        #self.redshift_tensor       = param_tensor[:, 6]             # shape [B]
        # Convert D_s from Mpc to kpc, then to tensor
        D_s_index = precomputed_dict["param_map"].index("D_s")
        self.D_s_tensor = precomputed_dict["params"][:, D_s_index] * 1000 #kpc

      
        self.std_rad_tensor = self.std_kpc_tensor / self.D_s_tensor  # shape [B]

        # Constant for exponent
        self.half = torch.tensor(0.5, device=device)
            
            
#         import json
#         self.debug_params = {
#             "position_rad":    self.position_rad_tensor.detach().cpu().numpy().tolist(),
#             "I":               self.I_tensor.detach().cpu().numpy().tolist(),
#             "orient_rad":      self.orient_rad_tensor.detach().cpu().numpy().tolist(),
#             "q":               self.q_tensor.detach().cpu().numpy().tolist(),
#             "std_kpc":         self.std_kpc_tensor.detach().cpu().numpy().tolist(),
#             "D_s_kpc":         self.D_s_tensor.detach().cpu().numpy().tolist(),
#             "std_rad":         self.std_rad_tensor.detach().cpu().numpy().tolist(),
#         }
        
#         print("batched")
#         print(json.dumps(self.debug_params, indent=4))
        
        
        # Buffers for batched operations
        self._buffers = {}
        self._batch_sizes_seen = set()
        self._rotation_factors = {}

    def _ensure_buffers(self, batch_size: int, height: int, width: int):
        """Allocate or resize all intermediate buffers"""
        key = (batch_size, height, width)
        if key in self._batch_sizes_seen:
            return
        # mark seen
        self._batch_sizes_seen.add(key)

        # Expand buffers
        self._buffers['pos_expanded'] = torch.empty((batch_size, height, width, 2), device=self.device)
        self._buffers['orient_expanded'] = torch.empty((batch_size, height, width), device=self.device)
        self._buffers['q_expanded'] = torch.empty((batch_size, height, width), device=self.device)
        self._buffers['std_expanded'] = torch.empty((batch_size, height, width), device=self.device)
        self._buffers['I_expanded'] = torch.empty((batch_size, height, width), device=self.device)

        # Buffers for rotation
        self._buffers['complex_z'] = torch.empty((batch_size, height, width), dtype=torch.complex64, device=self.device)
        self._buffers['rotated_z'] = torch.empty((batch_size, height, width), dtype=torch.complex64, device=self.device)

        # Buffers for final calc
        self._buffers['rs2'] = torch.empty((batch_size, height, width), device=self.device)
        self._buffers['sb'] = torch.empty((batch_size, height, width), device=self.device)

    def _get_rotation_factor(self, batch_size: int, orient_rad: torch.Tensor):
        """Cache and return exp(-i * theta) per batch"""
        if batch_size not in self._rotation_factors:
            self._rotation_factors[batch_size] = torch.empty((batch_size,), dtype=torch.cfloat, device=self.device)
        # compute in-place
        torch.exp((-1j * orient_rad).to(torch.cfloat), out=self._rotation_factors[batch_size])
        return self._rotation_factors[batch_size]

    def forward(self, source_grid: torch.Tensor) -> torch.Tensor:
        """
        Batched Gaussian blob surface-brightness in physical kpc units.

        Args:
            source_grid: Tensor of shape [B, H, W, 2] (positions in kpc)
        Returns:
            sb: Tensor of shape [B, H, W]
        """
        B, H, W, _ = source_grid.shape

        # 1) Translate & rotate each blob without any expand/copy
        #   source_grid   : (B,H,W,2)
        #   position      : (B,2) -> broadcast to (B,H,W,2)
        diff = source_grid - self.position_rad_tensor.view(B, 1, 1, 2)

        # view as complex (x + i y)
        #   diff.float() -> (B,H,W,2) -> view_as_complex -> (B,H,W)
        complex_z = torch.view_as_complex(
            diff.to(torch.float32)
                .view(B, H, W, 1, 2)
                .squeeze(-2)
        )

        # rotation factor already cached per batch
        rot = self._get_rotation_factor(B, self.orient_rad_tensor)  # (B,)

        # apply rotation: (B,H,W) * (B→B,1,1) → (B,H,W)
        rotated_z = complex_z * rot.view(B, 1, 1)

        # back to real coords: (B,H,W,2)
        xy = torch.view_as_real(rotated_z)

        # 2) Compute rs² = q²·x² + y², all in kpc²
        x, y = xy[..., 0], xy[..., 1]
        q      = self.q_tensor.view(B, 1, 1)     # (B→B,1,1)
        rs2    = q**2 * x*x + y*y                # → (B,H,W)

        # 3) Compute physical std in kpc and form the Gaussian
        std_rad = self.std_rad_tensor.view(B, 1, 1)      # [rad]
        exponent = -0.5 * rs2 / (std_rad * std_rad)


        sb       = torch.exp(exponent) * self.I_tensor.view(B, 1, 1)

        return sb  # shape (B, H, W)
