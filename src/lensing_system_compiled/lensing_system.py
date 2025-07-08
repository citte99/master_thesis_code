# lensing_system.py

import torch
import torch.nn as nn

class CatalogLensingSystem(nn.Module):
    """
    A single, catalog‐wide lensing + source pipeline.
    `forward(grid, batch_idx)` returns a [B,H,W] image stack
    for those `batch_idx` systems.
    """
    def __init__(self, masses_data: dict, precomputed_data: dict, source_data: dict,
                 device=None, dtype=torch.float32):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # ─── register all catalog tensors as buffers ───
        # mass components
        for mtype, dat in masses_data.items():
            self.register_buffer(f"{mtype}_params",   dat["params"])
            self.register_buffer(f"{mtype}_sys_idx",  dat["sys_idx"])
        # precomputed lookup
        self.register_buffer("precomp_params",  precomputed_data["params"])
        self.register_buffer("precomp_sys_idx", precomputed_data["sys_idx"])
        # encode param_map as a LongTensor for indexing
        self.register_buffer(
            "precomp_map",
            torch.tensor(
                [precomputed_data["param_map"].index(k)
                 for k in precomputed_data["param_map"]],
                dtype=torch.long
            )
        )
        # source components
        for stype, dat in source_data.items():
            self.register_buffer(f"{stype}_params",   dat["params"])
            self.register_buffer(f"{stype}_sys_idx",  dat["sys_idx"])

        # ─── static “factory” dicts ───
        from .lens_mass_components import PEMDStatic #SISStatic, NFWStatic, ExternalPotentialStatic, 
        self.component_classes = {
            # "SIS": SISStatic,
            # "NFW": NFWStatic,
            # "ExternalPotential": ExternalPotentialStatic,
            "PEMD": PEMDStatic,
        }
        from .sources import GaussianBlobStatic
        self.source_classes = {
            "Gaussian_blob": GaussianBlobStatic,
            # add more if you have them…
        }

    @torch.jit.export
    def forward(self, lens_grid: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
          lens_grid:  [H*W,2] or [1,H,W,2]  (we'll expand to [B,H,W,2])
          batch_idx:  LongTensor [B]
        Returns:
          brightness: [B, H, W]
        """
        B = batch_idx.size(0)
        # ensure right dtype/device
        grid = lens_grid.to(device=self.device, dtype=self.dtype)
        # expand to batch
        grid = grid.expand(B, *grid.shape[-3:])  # → [B,H,W,2]

        # ─── 1) deflection ───
        total_defl = torch.zeros_like(grid)
        for mtype, cls in self.component_classes.items():
            params  = getattr(self, f"{mtype}_params")
            sys_idx = getattr(self, f"{mtype}_sys_idx")
            # mask + slice
            mask    = (sys_idx.unsqueeze(1) == batch_idx.unsqueeze(0)).any(dim=1)
            p_sub   = params[mask]
            idx_sub = sys_idx[mask]
            # precomputed slice
            pre_p = self.precomp_params[mask]
            pre_i = self.precomp_sys_idx[mask]
            map_  = self.precomp_map
            # call static deflection
            defl = cls.deflection(p_sub, pre_p, pre_i, map_, grid)
            # remap system‐indices 0…B-1
            new_idx = (idx_sub.unsqueeze(1)==batch_idx.unsqueeze(0)).float().argmax(dim=1)
            total_defl.index_add_(0, new_idx, defl)

        source_grid = grid - total_defl

        # ─── 2) source brightness ───
        B,H,W,_ = source_grid.shape
        brightness = torch.zeros((B,H,W), device=self.device, dtype=self.dtype)
        for stype, cls in self.source_classes.items():
            params  = getattr(self, f"{stype}_params")
            sys_idx = getattr(self, f"{stype}_sys_idx")
            mask    = (sys_idx.unsqueeze(1) == batch_idx.unsqueeze(0)).any(dim=1)
            p_sub   = params[mask]
            idx_sub = sys_idx[mask]
            subgrid = source_grid[idx_sub]      # [M,H,W,2]
            vals    = cls.evaluate(p_sub, subgrid, self.precomp_params, self.precomp_map)
            # remap and sum
            new_idx = (idx_sub.unsqueeze(1)==batch_idx.unsqueeze(0)).float().argmax(dim=1)
            brightness.index_add_(0, new_idx, vals)

        return brightness
