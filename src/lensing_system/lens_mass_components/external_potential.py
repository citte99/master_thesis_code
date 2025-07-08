import torch

from .mass_component import MassComponent
from shared_utils import _arcsec_to_rad

class ExternalPotential(MassComponent):

    def __init__(self, config_dict, device):
        super().__init__()
        self.ss = config_dict["shear_strength"]
        self.sa = _arcsec_to_rad(config_dict["shear_angle_arcsec"])
        self.shear_center=config_dict["shear_center"]
        self. device = device
        
    def deflection_angle(self, lens_grid, precomp_dict=None, z_source=None):
        """
            precomp dict and z_source are there just for uniformity with the other mass components
        """
        xrel = lens_grid - self.shear_center
        cs2 = -torch.cos(2 * self.sa)
        sn2 = -torch.sin(2 * self.sa)
        alpha = torch.zeros_like(lens_grid, device=self.device)
        alpha[..., 0] = self.ss * (cs2 * xrel[..., 0] + sn2 * xrel[..., 1])
        alpha[..., 1] = self.ss * (sn2 * xrel[..., 0] - cs2 * xrel[..., 1])
        return alpha