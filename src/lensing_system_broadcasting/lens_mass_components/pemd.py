import torch
from torch import Tensor

from .mass_component import MassComponent
from .util import _hyp2f1_series, _hyp2f1_angular_series, _hyp2f1_angular_series_compiled #, scripted_hyp2f1_series

# create these only once
_TWO  = torch.tensor(2.0, dtype=torch.float32, device="cuda")

class PEMD(MassComponent):
    
    def __init__(self, param_tensor: torch.Tensor, precomputed_dict: dict, device="cuda:0", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.compute_dtype = torch.float32

        #ensure the tensor is on the right device and dtype



        # unpack
        self.slope     = param_tensor[:, 4]              # [B]
        self.pos       = param_tensor[:, 0:2]            # [B,2]
        self.th        = param_tensor[:, 5]              # [B]
        self.q         = param_tensor[:, 6]              # [B]
        self.vel_disp  = param_tensor[:, 3]              # [B]
        # convert distances from Mpc -> kpc

        key_map=precomputed_dict["param_map"]
        #get the index of D_s
        D_s_index  = key_map.index("D_s")
        D_ls_index = key_map.index("D_ls")
        Theta_E_index = key_map.index("Theta_E")

        self.D_s       = precomputed_dict["params"][:, D_s_index] * 1000       # [B]
        self.D_ls      = precomputed_dict["params"][:, D_ls_index] * 1000       # [B]
        self.Theta_E   = precomputed_dict["params"][:, Theta_E_index]       # [B]

        # Pre-compute constants in compute precision

        self.two  = _TWO


    # -------------------------------------------------------------------
    # deflection angle (Tessore & Metcalf 2015, § 4) – fixed rotation
    # -------------------------------------------------------------------
    def deflection_angle(self, lens_grid: torch.Tensor) -> torch.Tensor:

        # ---- promote precision ----------------------------------------
        grid = lens_grid.to(self.compute_dtype, copy=False)

        # ---- constants & 1×1×1 tensors --------------------------------
        theta_E = self.Theta_E.view(-1, 1, 1)
        q       = self.q.view(-1, 1, 1)
        t       = self.slope.view(-1, 1, 1)
        th      = self.th.view(-1, 1, 1)
        b       = theta_E * torch.sqrt(q)

        f   = (1.0 - q) / (1.0 + q)          # (1−q)/(1+q)  < 1
        two = torch.tensor(2.0,
                        dtype=self.compute_dtype,
                        device=lens_grid.device)

        # ---- shift & rotate coordinates -------------------------------
        pos  = self.pos.view(-1, 1, 1, 2)
        diff = grid - pos
        zc   = torch.view_as_complex(diff)        # x + i y  in sky frame
        crot = torch.exp(-1j * th)                # e^{-iθ}
        zc   = crot * zc                          # into lens frame

        x_p, y_p = zc.real, zc.imag

        # ---- elliptical radius & phase --------------------------------
        R2   = q**2 * x_p**2 + y_p**2
        R    = torch.sqrt(R2)
        eps  = torch.finfo(R.dtype).eps
        R    = torch.where(R > eps, R, eps)

        exp_iφ  = (q * x_p + 1j * y_p) / R           # unit complex
        exp_2iφ = exp_iφ ** 2

        # ---- angular hypergeometric argument --------------------------
        z_ang = -f * exp_2iφ                         # |z_ang| = f < 1

        # ---- angular series -------------------------------------------
        F_ang = _hyp2f1_angular_series_compiled(z_ang, t, max_terms=15)

        # ---- analytic prefactor  (eq. 29 TM15) ------------------------
        pref   = (two * b) / (1.0 + q)
        A_ang  = pref * (b / R) ** (t - 1.0) * exp_iφ

        # ---- lens-frame deflection ------------------------------------
        alpha_lens = A_ang * F_ang                  # α(R,φ)

        # ---- rotate back to sky frame  (multiply by e^{+iθ}) ----------
        alpha_c = alpha_lens * torch.conj(crot)     # NO conjugation of α!

        # ---- split into (α_x, α_y) ------------------------------------
        alpha = torch.view_as_real(alpha_c)

        # ---- cast back to requested dtype -----------------------------
        return alpha.to(self.dtype) if self.dtype != self.compute_dtype else alpha

    
    
    
#      def deflection_angle_non_tessore_but_devon(self, lens_grid):
#         """
#         Compute the batched deflection angle for the PEMD profile.
        
#         Parameters:
#           lens_grid: Tensor of shape [B, H, W, 2] representing the angular grid
#                      in the lens plane.
#           precomp_dict: Dummy argument (distances are provided in the input tensor).
#           z_source: Dummy argument.
          
#         Returns:
#           A tensor of deflection angles of shape [B, H, W, 2].
#         """
#         # Upcast input grid if needed
#         lens_grid_compute = lens_grid if lens_grid.dtype == self.compute_dtype else lens_grid.to(dtype=self.compute_dtype)
        
              
#         # Compute Einstein radius more efficiently
#         theta_E = self.Theta_E.view(-1, 1, 1)  # [B, 1, 1] in radians
        
#         # Compute the scale parameter b.
#         q = self.q.view(-1, 1, 1)  # [B, 1, 1]
#         b = theta_E * torch.sqrt(q)  # [B, 1, 1]
        
#         # Prepare the slope parameter for broadcasting.
#         t = self.slope.view(-1, 1, 1)  # [B, 1, 1]
        
#         # Rotate coordinates:
#         pos = self.pos.view(-1, 1, 1, 2)  # [B, 1, 1, 2]
#         diff = lens_grid_compute - pos    # [B, H, W, 2]


#         z = torch.view_as_complex(diff)  # [B, H, W]
        
#         # Apply orientation rotation.
#         th = self.th.view(-1, 1, 1)       # [B, 1, 1]
#         crot = torch.exp(-1j * th)
#         z = crot * z  # rotated coordinates
        
#         # Compute the elliptical coordinate:
#         rs2 = (q**2) * (z.real**2) + (z.imag**2)  # [B, H, W]
#         rs = torch.sqrt(rs2)
        
#         # Compute factor A = b² / (q * z) * (b / rs)^(t - 2)
#         A = (b**2) / (q * z) * (b / rs)**(t - self.two)
        
#         # Compute the hypergeometric series term.
#         #F = _hyp2f1_series(z, rs2, t, q)
#         F = _hyp2f1_series(z, rs2, t, q)

        
#         # Compute the complex deflection angle and rotate back.
#         # Add resolve_conj() to fix the "unresolved conjugated tensors" error
#         alpha_complex = torch.conj(A * F * crot).resolve_conj()
        
#         # Convert the complex deflection to a real 2-vector field.
#         alpha_real = torch.view_as_real(alpha_complex)
        
#         # Convert to target precision only at the end if needed
#         if self.dtype != self.compute_dtype:
#             result = alpha_real.to(dtype=self.dtype)
#         else:
#             result = alpha_real
            
#         return result
    