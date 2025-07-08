import torch

from .mass_component import MassComponent
import shared_utils.units as units

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEFAULT_DEVICE)
DEFAULT_DTYPE  = torch.float32

# Static constants as tensors on default device/dtype
CONST       = torch.tensor(2.16258, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
PI          = torch.tensor(torch.pi, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
G           = torch.tensor(units.G, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
C_SQUARED   = torch.tensor(units.c**2, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
EPSILON     = torch.tensor(1e-8, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
FOUR        = torch.tensor(4.0, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
TWO         = torch.tensor(2.0, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
ONE         = torch.tensor(1.0, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
ZERO        = torch.tensor(0.0, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

class NFW(MassComponent):
    """
    Navarro–Frenk–White (NFW) profile, batched.

    Expects param_tensor of shape [B, 7] with columns:
    [x, y, mass_max, r_max_kpc, D_l, D_s, D_ls] (distances in Mpc)
    """
    def __init__(self, param_tensor: torch.Tensor, precomputed_dict: dict, device="cuda", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)


        # unpack columns
        self.pos        = param_tensor[:, :2]            # [B, 2]
        self.mass_max   = param_tensor[:, 2]             # [B]
        self.r_max_kpc  = param_tensor[:, 3]             # [B]
        # convert distances from Mpc -> kpc
    
        key_map=precomputed_dict["param_map"]
        #get the index of D_s
        D_s_index  = key_map.index("D_s")
        D_ls_index = key_map.index("D_ls")
        D_l_index = key_map.index("D_l")


        self.D_s       = precomputed_dict["params"][:, D_s_index] * 1000       # [B]
        self.D_ls      = precomputed_dict["params"][:, D_ls_index] * 1000       # [B]
        self.D_l      = precomputed_dict["params"][:, D_l_index] * 1000       # [B]

        


        # Constants in compute precision
        self.const      = CONST
        self.pi         = PI
        self.G          = G
        self.c_squared  = C_SQUARED
        self.epsilon    = EPSILON
        self.four       = FOUR
        self.two        = TWO
        self.one        = ONE
        self.zero       = ZERO
        # Pre-compute derivable parameters (all in compute precision)
        # r_s calculation
        self.r_s = self.r_max_kpc / self.const
        
        # rho_s calculation
        log_term = torch.log(1.0 + self.const)
        const_term = self.const / (1.0 + self.const)
        denominator = log_term - const_term
        r_s_cubed = self.r_s ** 3
        numerator = self.mass_max / (self.four * self.pi * r_s_cubed)
        self.rho_s = numerator / denominator
        
        # Pre-compute sigma_crit
        num = self.c_squared * self.D_s
        denom = self.four * self.pi * self.G * self.D_l * self.D_ls
        self.sigma_crit = num / denom
        
        # Pre-compute kappa_s
        self.ks = (self.r_s * self.rho_s) / self.sigma_crit
        
        # Pre-compute partial factor for alpha calculation (the part that doesn't depend on the grid)
        self.alpha_factor_partial = self.four * self.ks * self.r_s
        
        # Buffers will be allocated on first use
        self._initialized_buffers = False
        self._buffer_shapes = None
        
    
    def _initialize_buffers(self, batch_size, height, width):
        """Initialize all buffers at once with compute precision"""
        # Only initialize once for each shape configuration
        current_shapes = (batch_size, height, width)
        if self._initialized_buffers and self._buffer_shapes == current_shapes:
            return
        
        grid_shape = (batch_size, height, width)
        
        # Create all buffers in compute precision directly
        self._xrel = torch.empty((batch_size, height, width, 2), 
                            device=self.device)
        self._rs2 = torch.empty(grid_shape, 
                            device=self.device)
        self._rs = torch.empty(grid_shape, 
                            device=self.device)
        self._rs_nodim = torch.empty(grid_shape, 
                                device=self.device)
        self._F = torch.empty(grid_shape, 
                            device=self.device)
        self._log_term = torch.empty(grid_shape, 
                                device=self.device)
        self._alpha = torch.empty(grid_shape, 
                                device=self.device)
        self._alpha_vec = torch.empty((batch_size, height, width, 2), 
                                device=self.device)
        
        self._initialized_buffers = True
        self._buffer_shapes = current_shapes
    
    def deflection_angle(self, lens_grid, z_source=None):
        """
        lens_grid : Tensor of shape (H,W,2) or (B,H,W,2)
        returns   : Tensor of shape (B,H,W,2), broadcasting the single grid
                    to all B halos without any explicit expand/copy.
        """
        # 1) Force correct device/dtype
        lens_grid = lens_grid.to(device=self.device, dtype=self.dtype)

        # 2) Figure out B,H,W from input
        if lens_grid.dim() == 3:
            # single grid for all halos
            B = self.pos.shape[0]
            H, W, _ = lens_grid.shape
        elif lens_grid.dim() == 4:
            B, H, W, _ = lens_grid.shape
            if B != self.pos.shape[0]:
                raise ValueError(f"Batch-size mismatch: lens_grid has {B}, but NFW has {self.pos.shape[0]}")
        else:
            raise ValueError(f"lens_grid must be 3D or 4D, got {lens_grid.dim()}D")

        # 3) (Re)allocate your buffers once to (B,H,W,…)
        self._initialize_buffers(B, H, W)

        # 4) Compute relative positions via broadcasted subtraction:
        pos_expanded = self.pos.view(B, 1, 1, 2)        # (B,1,1,2)
        torch.sub(lens_grid, pos_expanded, out=self._xrel)  # -> (B,H,W,2)

        # 5) radius² = x² + y² (in-place)
        x = self._xrel[..., 0]
        y = self._xrel[..., 1]
        torch.mul(x, x, out=self._rs2)
        torch.addcmul(self._rs2, y, y, value=1.0, out=self._rs2)

        # 6) radius = sqrt(rs2) with epsilon floor
        torch.sqrt(self._rs2, out=self._rs)
        torch.maximum(self._rs, self.epsilon, out=self._rs)

        # 7) dimensionless radius = (rs * D_l) / r_s
        D_l_exp = self.D_l.view(-1, 1, 1)               # (B,1,1)
        r_s_exp = self.r_s.view(-1, 1, 1)               # (B,1,1)
        torch.mul(self._rs,     D_l_exp, out=self._rs_nodim)
        torch.div(self._rs_nodim, r_s_exp, out=self._rs_nodim)

        # 8) compute F(rs_nodim) in one tensor
        sq = self._rs_nodim * self._rs_nodim
        t1 = torch.sqrt(torch.clamp(1 - sq,     min=self.epsilon))
        t3 = torch.sqrt(torch.clamp(sq - 1,     min=self.epsilon))
        F1 = torch.atanh(t1) / t1
        F3 = torch.atan(t3)  / t3
        cond1 = self._rs_nodim < 1
        cond3 = self._rs_nodim > 1
        self._F = torch.where(cond1, F1,
                    torch.where(cond3, F3,
                                torch.ones_like(self._rs_nodim)))

        # 9) log term = log(rs_nodim / 2)
        torch.div(self._rs_nodim, self.two, out=self._log_term)
        torch.log(self._log_term,      out=self._log_term)

        # 10) deflection magnitude α
        alpha_fp = self.alpha_factor_partial.view(-1,1,1)  # (B,1,1)
        denom    = D_l_exp * self._rs_nodim
        alpha_f  = alpha_fp / denom
        torch.add(self._log_term, self._F,  out=self._alpha)
        torch.mul(self._alpha,     alpha_f, out=self._alpha)

        # 11) deflection vector = α * (xrel / rs)
        a = self._alpha.view(B, H, W, 1)                 # (B,H,W,1)
        r = self._rs.view(B, H, W, 1)                    # <— define r here!
        torch.mul(a, self._xrel,     out=self._alpha_vec)
        torch.div(self._alpha_vec, r, out=self._alpha_vec)

        return self._alpha_vec  # (B,H,W,2)

