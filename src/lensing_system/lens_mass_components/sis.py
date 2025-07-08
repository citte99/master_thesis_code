import torch

from .mass_component import MassComponent
import shared_utils.units as units

class SIS (MassComponent):
    def __init__(self, config_dict, device):
        """
            We unpack the tensor configuration in the common features
            between different mass components.

            In this code, all the components are at same redshift, so the
            redshift is not a parameter of the mass component.

            We have only the position.
        """
        super().__init__()
        self.device = device
        self.pos= config_dict["pos"]
        self.redshift = config_dict["redshift"]
        self.vel_disp = config_dict["vel_disp"]
        # print(self.pos.device)
        # print(torch.device(self.device))
        # assert str(self.pos.device) == str(self.device), "Tensor is on the wrong device!"
        # assert self.redshift.device == self.device, "Tensor is on the wrong device!"
        # assert self.vel_disp.device == self.device, "Tensor is on the wrong device!"


        self.c=torch.as_tensor(units.c).to(self.device)
        # assert self.c.device == torch.device(self.device), "Tensor is on the wrong device!"
        

    pass

    
    def deflection_angle(self, lens_grid, precomp_dict=None, z_source=None):
        """
        This computes the deflection field for the SIS model.
        z_source is a dummy here the distances D_s and D_ls are provided via precomp_dict.
        """
        assert z_source is None, "z_source should be passed implicitly in precomp_dict. Dummy argument for clarity."

        # Ensure distances are torch tensors on the proper device
        D_s = precomp_dict["D_s"]
        D_ls = precomp_dict["D_ls"]

        self.einstein_angle = 4 * torch.pi * (self.vel_disp / self.c)**2 * D_ls / D_s

        # Compute the relative position from the lens center.
        x_rel = lens_grid - self.pos

        # Sum across the last dimension to compute the radial distance.
        r = torch.sqrt(torch.sum(x_rel**2, dim=-1))
        epsilon = 1e-8
        # Replace zeros to avoid division by zero.
        r = torch.where(r == 0, torch.tensor(epsilon, device=self.device), r)
        # Unsqueeze to broadcast correctly.
        r = r.unsqueeze(-1)
        
        return self.einstein_angle * x_rel / r
