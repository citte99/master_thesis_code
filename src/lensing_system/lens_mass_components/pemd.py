import torch

from .mass_component import MassComponent
from .util import _hyp2f1_series, _hyp2f1_angular_series
from shared_utils import units

class PEMD(MassComponent):
    """
        Power law elliptical mass distribution
    """

    def __init__(self, config_dict, device):
        super().__init__()
        self.device = device

        self.slope = config_dict["slope"]   
        self.pos = config_dict["pos"]
        self.th = config_dict["orient"]
        self.q = config_dict["q"]
        self.vel_disp = config_dict["vel_disp"]


#     def deflection_angle(self, lens_grid, precomp_dict=None, z_source=None):
#         '''
#             z_source is a dummy here, the distances D_s and D_ls are provided via precomp_dict.

    
#         This procedure matches exactly equation 13 from 
#         https://arxiv.org/pdf/1507.01819

#         The mapping is as follows:
#         if the convergence is k(R)=(2-t)/2 * (b/R)^t and R^2=q^2x^2+y^2
#         then 
#         HERE   vs   PAPER
#         rs     R
#         rs2    R^2
#         q      q
#         t      t
#         b      b

#         The question remains on what is self.qh, and what is self.b (which should be the einstein radius
#         (in parsec right? from my calculations I have b=\theta_ein*D_l))

#         '''
#         self.b=precomp_dict["Theta_E"]
#         t = self.slope
#         q = self.q
#         #I guess general relation between einstein radius and scale paramter b of the power law
#         #NOTE: question!!!!!!!!!
#         #b = (0.5 * self.b * q**(t - 0.5) * (3.0 - t) / (2.0 - t))**(1.0 / t)
#         #the following is from cono'r code
#         b=self.b*torch.sqrt(q)
#         #here just rotating the coordinates to move to a frame where the lens is aligned with the x axis
#         crot = torch.exp(-1j * (self.th)) #before we had theta_to_rad(self.th), but now its in rad 
#         z = crot * torch.view_as_complex(lens_grid - self.pos)
#         rs2 = (q**2 * z.real**2 + z.imag**2)
#         rs = torch.sqrt(rs2)
#         A = b**2 / (q * z) * (b / rs)**(t - 2)
#         F = _hyp2f1_series(z, rs2, t, q)
#         alpha = torch.conj(A * F * crot).resolve_conj()
#         return torch.view_as_real(alpha)

    
    def deflection_angle(self, lens_grid, precomp_dict=None, z_source=None):
        """
        Compute the PEMD deflection using the angular-series expansion (eq. 13 of Tessore & Metcalf 2015),
        without any extra batch broadcasting. Assumes:
          - lens_grid: FloatTensor of shape (H, W, 2)
          - self.pos: FloatTensor of shape (2,)
          - precomp_dict["Theta_E"]: scalar Tensor
          - self.slope, self.q, self.th: scalar Tensors
        Returns:
          - alpha: FloatTensor of shape (H, W, 2)
        """
        # unpack scalars
        theta_E = 4 * torch.pi * (self.vel_disp / units.c)**2 * precomp_dict["D_ls"] / precomp_dict["D_s"]      # scalar
        
        print(theta_E)
        t       = self.slope                   # scalar
        q       = self.q                       # scalar
        th      = self.th                      # scalar

        # scale and ellipticity
        b = theta_E * torch.sqrt(q)
        f = (1.0 - q) / (1.0 + q)

        # rotate coords into lens frame
        # `pos` is (2,), lens_grid is (H,W,2) → complex z of shape (H,W)
        crot = torch.exp(-1j * th)
        z    = crot * torch.view_as_complex(lens_grid - self.pos)

        x, y = z.real, z.imag

        # elliptical radius
        rs2 = q**2 * x**2 + y**2
        rs  = torch.sqrt(rs2)

        # avoid singularity at r=0
        eps = torch.finfo(rs.dtype).eps
        rs  = torch.where(rs > eps, rs, eps)

        # phase factors
        exp_iφ  = (q * x + 1j * y) / rs
        exp_2iφ = exp_iφ**2

        # angular series
        z_ang = -f * exp_2iφ
        F_ang = _hyp2f1_angular_series(z_ang, t, max_terms=15)

        # analytic prefactor (eq. 29)
        pref  = (2.0 * b) / (1.0 + q)
        A_ang = pref * (b / rs)**(t - 1.0) * exp_iφ

        # deflection (lens frame → sky frame)
        alpha_c = A_ang * F_ang * torch.conj(crot)

        # split into real vector field (H, W, 2)
        return torch.view_as_real(alpha_c)


