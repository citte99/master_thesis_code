import numpy as np
import torch
from .source_model import SourceModel
from .util import _translate_rotate

__all__ = [
    "SersicClumps",
    "generate_sersic_clumps_config",
]

# =============================================================================
#  Helper ---------------------------------------------------------------------
# =============================================================================

def _to_tensor(x, device):
    """Convert *x*—possibly a (nested) list/tuple, NumPy array or tensor—into a
    **single** `torch.Tensor` on *device* (float32).  Handles the case where
    JSON‑deserialised lists have already been partially converted to 0‑D
    tensors by upstream helpers.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device).float()

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return torch.empty(0, device=device)

        if isinstance(x[0], torch.Tensor):  # list of tensors → stack
            return torch.stack([t.to(device).float() for t in x])

        if isinstance(x[0], (list, tuple)):
            return torch.stack([_to_tensor(sub, device) for sub in x])

        return torch.as_tensor(x, device=device, dtype=torch.float32)

    return torch.as_tensor(x, device=device, dtype=torch.float32)


# =============================================================================
#  Sérsic‑clump source model ---------------------------------------------------
# =============================================================================

class SersicClumps(SourceModel):
    """A clumpy extended source built from multiple Sérsic ellipses.

    **Parameter naming change (May‑2025)**
    -------------------------------------
    * ``position_rad``               – (2,)   global centre of the *whole* source.
    * ``relative_pos_single_blobs``  – (N,2) per‑clump offsets *relative* to that
                                        global centre.

    Legacy configs that still pass a key ``positions_rad`` (absolute clump
    centres) are accepted for backward compatibility.
    """

    def __init__(self, config_dict, precomp_dict, device):
        super().__init__()
        self.device = device

        # ------------------------------------------------------------------
        #  Required parameters (all pre‑sampled)
        # ------------------------------------------------------------------
        self.I           = _to_tensor(config_dict["I"], device)
        self.R_ser_kpc   = _to_tensor(config_dict["R_ser_kpc"], device)
        self.n           = _to_tensor(config_dict["n"], device)
        self.e           = _to_tensor(config_dict["ellipticity"], device)

        # ----- positions ---------------------------------------------------
        if "relative_pos_single_blobs" in config_dict:
            rel = _to_tensor(config_dict["relative_pos_single_blobs"], device)
            cen = _to_tensor(config_dict.get("position_rad", [0.0, 0.0]), device)
            if cen.ndim == 1:
                cen = cen.unsqueeze(0)          # (1,2) → broadcastable
            self.positions = rel + cen          # absolute positions (N,2)
        else:
            # Fallback to old "positions_rad" key (already absolute)
            self.positions = _to_tensor(config_dict["positions_rad"], device)

        # ------------------------------------------------------------------
        #  Derived quantities
        # ------------------------------------------------------------------
        self.D_s       = precomp_dict["D_s"] * 1000.0          # Mpc → kpc
        self.R_ser_rad = self.R_ser_kpc / self.D_s            # (N,)

        e_mag = torch.clamp(torch.sqrt((self.e ** 2).sum(-1)), max=0.999)
        self.phi = 0.5 * torch.atan2(self.e[..., 1], self.e[..., 0])  # (N,)
        self.q   = (1.0 - e_mag) / (1.0 + e_mag)                      # (N,)

        self.b_n = 1.9992 * self.n - 0.3271  # Sérsic constant per clump

    # ----------------------------------------------------------------------
    def _sersic_profile(self, r, idx):
        bn = self.b_n[idx]
        n  = self.n[idx]
        Re = self.R_ser_rad[idx]
        I0 = self.I[idx]
        return I0 * torch.exp(-bn * (((r / Re) + 1e-12) ** (1.0 / n) - 1.0))

    # ----------------------------------------------------------------------
    def forward(self, source_grid):
        """Return total surface brightness evaluated on *source_grid* (…,2)."""
        sb = torch.zeros(source_grid.shape[:-1], device=self.device)
        for i in range(self.I.shape[0]):
            
            xy_rel = _translate_rotate(source_grid, self.positions[i], th_rad=self.phi[i])
            r2 = xy_rel[..., 0] ** 2 + (xy_rel[..., 1] / self.q[i]) ** 2
            r  = torch.sqrt(r2 + 1e-12)
            sb = sb + self._sersic_profile(r, i)
        self.surface_brightness = sb
        return sb


