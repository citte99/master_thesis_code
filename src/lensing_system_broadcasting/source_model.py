import torch
import torch.nn as nn
from .sources import GaussianBlob #Sersic  # add other source classes as needed

class SourceModel(nn.Module):
    """
    Batched source model using prestructured flat-catalog data.
    """
    def __init__(self, source_data: dict, precomputed: dict, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Map from source type key to the source class implementation
        self.source_classes = {
            "Gaussian_blob": GaussianBlob,
        #    "Sersic": Sersic,
            # add other source types here
        }

        # Instantiate one component per source type
        self.components = {}
        self.system_indices = {}
        for stype, dat in source_data.items():
            if stype not in self.source_classes:
                raise ValueError(f"Unknown source model type: {stype}")
            # params tensor has shape [M, K]
            params = dat['params']
            # which system each row belongs to
            idxs   = dat['sys_idx']

            # instantiate the source class; each should accept (param_tensor, precomputed, device, dtype)
            cls = self.source_classes[stype]
            self.components[stype] = cls(params, precomputed)
            self.system_indices[stype] = idxs

    def forward(self, source_grid: torch.Tensor) -> torch.Tensor:
        """
        Evaluate and sum all source contributions on the source_grid.

        Args:
            source_grid: [B, H, W, 2]

        Returns:
            brightness: Tensor of shape [B, H, W]
        """
        if source_grid.dtype != self.dtype:
            source_grid = source_grid.to(dtype=self.dtype)

        B, H, W, _ = source_grid.shape
        output = torch.zeros((B, H, W), device=self.device, dtype=self.dtype)

        # loop over each source type
        for stype, component in self.components.items():
            idxs = self.system_indices[stype]
            subgrid = source_grid[idxs]        # [M, H, W, 2]
            vals    = component(subgrid)       # [M, H, W]
            output.index_add_(0, idxs, vals)

        return output