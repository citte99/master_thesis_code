'''sources.py: static, pure-Torch Gaussian blob evaluation'''  
import torch

class GaussianBlobStatic:
    @staticmethod
    def evaluate(param_tensor: torch.Tensor,
                 source_grid: torch.Tensor,
                 precomp_params: torch.Tensor,
                 precomp_map: list) -> torch.Tensor:
        """
        Batched surface-brightness for a Gaussian blob source model (static).

        Args:
            param_tensor: [M, K] parameters: [x, y, I, orient, q, std_kpc, ...]
            source_grid: [M, H, W, 2] positions in radians
            precomp_params: [M, P] precomputed cosmological distances
            precomp_map: list of length P, names of each precomp_params column
        Returns:
            sb: [M, H, W] surface brightness
        """
        # unpack parameters
        pos     = param_tensor[:, 0:2].view(-1, 1, 1, 2)
        I       = param_tensor[:, 2].view(-1, 1, 1)
        orient  = param_tensor[:, 3].view(-1)
        q       = param_tensor[:, 4].view(-1, 1, 1)
        std_kpc = param_tensor[:, 5].view(-1)

        # find D_s index
        D_s_idx = precomp_map.index("D_s")
        D_s = precomp_params[:, D_s_idx] * 1000.0  # Mpc->kpc

        # physical std rad
        std_rad = (std_kpc / D_s).view(-1, 1, 1)

        # translate
        diff = source_grid - pos  # shape [M,H,W,2]
        # view as complex
        z = torch.view_as_complex(diff.to(torch.float32))  # [M,H,W]

        # rotation factor
        rot = torch.exp(-1j * orient).view(-1, 1, 1)  # [M,1,1]
        z_rot = z * rot

        # back to real coords
        xy = torch.view_as_real(z_rot)  # [M,H,W,2]
        x, y = xy[..., 0], xy[..., 1]

        # elliptical radius squared in kpc^2
        rs2 = q * q * x * x + y * y

        # exponent
        exponent = -0.5 * rs2 / (std_rad * std_rad)

        sb = torch.exp(exponent) * I
        return sb
