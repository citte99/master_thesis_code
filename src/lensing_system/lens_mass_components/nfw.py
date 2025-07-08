import torch

from .mass_component import MassComponent
import shared_utils.units as units


class NFW(MassComponent):

    #def __init__(self, pos_rel_main, mass_max, main_lens, z_source, r_max=False, *args, **kwargs):
    def __init__(self, config_dict, device):
        """
            For the NFW profile, mass_max and r_max can be related.
            This relation is handled at a higher level, here we just accept both as input.
        """
        super().__init__()
        self.device = device
        self.pos= config_dict["pos"]
        #self.redshift = config_dict["redshift"] #Not needed here, using the distances in precomp_dict
        self.mass_max = config_dict["mass_max"]
        self.r_max_kpc = config_dict["r_max_kpc"]
        # assert self.pos.device == torch.device(self.device), "Tensor is on the wrong device!"
        #assert self.redshift.device == torch.device(self.device), "Tensor is on the wrong device!"
        # assert self.mass_max.device == torch.device(self.device), "Tensor is on the wrong device!"
        # assert self.r_max_kpc.device == torch.device(self.device), "Tensor is on the wrong device!"

        #our form for the deflection angle depends on r_s and rho_s
        self.const=torch.tensor(2.16258, device=self.device)
        self.r_s=self.r_max_kpc/self.const #in kpc

        self.pi=torch.tensor(torch.pi, device=self.device)
        self.rho_s=self.mass_max/(4.0*self.pi*self.r_s**3)/(torch.log(1.0+self.const)-self.const/(1+self.const))#in M_sun/kpc^3
      

    def deflection_angle(self, lens_grid, precomp_dict=None, z_source=None):
        """
            z_source is a dummy here, the distances D_s and D_ls are provided via precomp_dict.

        """
        #despali 2018
        #https://arxiv.org/pdf/1710.05029
        #https://www.aanda.org/articles/aa/pdf/2013/12/aa21618-13.pdf
        assert z_source is None, "z_source should be passed implicitly in precomp_dict. Dummy argument for clarity."
        # assert lens_grid.device==self.device, "NFW object and lens_grid on different devices!"
        
        D_l=precomp_dict["D_l"]*1000 #in kpc
        D_s=precomp_dict["D_s"]*1000 #in kpc
        D_ls=precomp_dict["D_ls"]*1000 #in kpc
        # assert D_l.device == torch.device(self.device), "Tensor is on the wrong device!"
        # assert D_s.device == torch.device(self.device), "Tensor is on the wrong device!"
        # assert D_ls.device == torch.device(self.device), "Tensor is on the wrong device!"
        self.G=torch.tensor(units.G, device=self.device)
        sigma_crit=units.c**2/(4*self.pi*self.G)*D_s/(D_l*D_ls) #in M_sun/kpc^2
        ks=self.r_s*self.rho_s/sigma_crit

        xrel = lens_grid - self.pos
        rs2 = xrel[...,0]**2 + xrel[...,1]**2
        rs = torch.sqrt(rs2)

        # follow https://www.aanda.org/articles/aa/pdf/2013/12/aa21618-13.pdf for this rescaling
        #rs are the angles, 
        #map them to distances,
        #divide them by the scale r_s
        rs_nodim=rs*D_l/(self.r_s) #they all should be in kiloparsec


        alpha = torch.zeros_like(rs_nodim, device=self.device)
        F=torch.zeros_like(rs_nodim, device=self.device)
        mask1 = rs_nodim < 1
        mask2 = rs_nodim == 1.0
        mask3 = rs_nodim > 1

        F[mask1]=1/torch.sqrt(1-rs_nodim[mask1]**2)*torch.atanh(torch.sqrt(1-rs_nodim[mask1]**2))
        F[mask2]=1
        F[mask3]=1/torch.sqrt(rs_nodim[mask3]**2-1)*torch.atan(torch.sqrt(rs_nodim[mask3]**2-1))

        alpha=4*ks/rs_nodim*self.r_s/D_l*(torch.log(rs_nodim/2)+F) #is there a 2 missing before the F?

        #now returning the vector deflection angle
        alpha_x=alpha*xrel[...,0]/rs #there is a multiplication and then division by rs that could be avoided, but it is clearer this way
        alpha_y=alpha*xrel[...,1]/rs
        return torch.stack((alpha_x, alpha_y), dim=-1)
