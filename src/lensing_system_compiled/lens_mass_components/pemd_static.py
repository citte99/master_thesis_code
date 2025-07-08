import torch
from .util import _hyp2f1_series

class PEMDStatic:
    @staticmethod
    def deflection(param_tensor: torch.Tensor,
                   precomp_params: torch.Tensor,
                   precomp_map: list,
                   lens_grid: torch.Tensor) -> torch.Tensor:
        """
        Batched deflection for a PEMD profile (static, functional).

        Args:
            param_tensor: [M, K] tensor of PEMD parameters.
            precomp_params: [M, P] tensor of precomputed distances and scales.
            precomp_map: list of length P, names of each precomp_params column.
            lens_grid: [B, H, W, 2] grid of angular positions in radians.
        Returns:
            alpha: [B, H, W, 2] deflection vectors in the same units as lens_grid.
        """
        # parameter unpacking
        # param columns: 0-1 pos (rad), 3 vel_disp, 4 slope t, 5 th orientation, 6 q axis ratio
        pos   = param_tensor[:, 0:2].view(-1, 1, 1, 2)
        vel   = param_tensor[:, 3].view(-1, 1, 1)      # unused
        t     = param_tensor[:, 4].view(-1, 1, 1)      # slope
        th    = param_tensor[:, 5].view(-1, 1, 1)      # orientation
        q     = param_tensor[:, 6].view(-1, 1, 1)      # axis ratio

        # precomputed distances
        # find indices in precomp_map
        Theta_E_idx = precomp_map.index("Theta_E")
        D_s_idx     = precomp_map.index("D_s")
        #D_ls_idx    = precomp_map.index("D_ls")  # unused here

        Theta_E = precomp_params[:, Theta_E_idx].view(-1, 1, 1)
        D_s     = precomp_params[:, D_s_idx].view(-1, 1, 1) * 1000.0  # convert Mpc -> kpc

        # shift to compute deflection
        # ensure compute dtype is float32
        grid = lens_grid.to(dtype=torch.float32)

        # rotation to align major axis
        diff = grid - pos  # [B,M,H,W,2] via broadcasting
        z = torch.view_as_complex(diff)
        rot = torch.exp(-1j * th.squeeze(-1).squeeze(-1))  # [M]
        z_rot = z * rot.view(-1, 1, 1)

        # elliptical radius
        rs2 = (q**2) * (z_rot.real**2) + (z_rot.imag**2)
        rs = torch.sqrt(rs2)

        # scale b = Theta_E * sqrt(q)
        b = Theta_E * torch.sqrt(q)
        # constant two for exponent
        two = torch.tensor(2.0, device=z.device, dtype=z.dtype)

        # factor A = b^2 / (q * z) * (b/rs)^(t-2)
        A = (b**2) / (q * z_rot) * (b / rs)**(t - two)

        # hypergeometric series term F
        F = _hyp2f1_series(z_rot, rs2, t, q)

        # complex deflection angle, rotated back
        alpha_complex = torch.conj(A * F * rot.conj()).resolve_conj()
        alpha = torch.view_as_real(alpha_complex)  # [M, H, W, 2]

        # cast back to input dtype if needed
        if alpha.dtype != lens_grid.dtype:
            alpha = alpha.to(dtype=lens_grid.dtype)

        return alpha

