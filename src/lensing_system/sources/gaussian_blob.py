import torch

from .source_model import SourceModel
from .util import _translate_rotate

class GaussianBlob(SourceModel):
    """
    """
    def __init__(self, config_dict, precomp_dict, device):
        super().__init__()
        self.device = device
        self.I = config_dict["I"]
        self.position = config_dict["position_rad"]
        self.orient_rad = config_dict["orient_rad"]
        self.q = config_dict["q"]
        self.std_kpc = config_dict["std_kpc"]
        self.D_s = precomp_dict["D_s"] * 1000.  # Mpc to kpc
        self.std_rad = self.std_kpc / self.D_s

#         import json
#          # collect into debug dict
#         self.debug_params = {
#             "position_rad":    self.position.detach().cpu().numpy().tolist(),
#             "I":               self.I.detach().cpu().numpy().tolist(),
#             "orient_rad":      self.orient_rad.detach().cpu().numpy().tolist(),
#             "q":               self.q.detach().cpu().numpy().tolist(),
#             "std_kpc":         self.std_kpc.detach().cpu().numpy().tolist(),
#             "D_s_kpc":         self.D_s.detach().cpu().numpy().tolist(),
#             "std_rad":         self.std_rad.detach().cpu().numpy().tolist(),
#         }
#         print("non_batched")
#         print(json.dumps(self.debug_params, indent=4))
        
        

    def forward(self, source_grid):
        #use a torch method to translate and rotate the grid


        xrel = _translate_rotate(source_grid, self.position, th_rad=self.orient_rad)
        rs2 = self.q**2 * xrel[..., 0]**2 + xrel[..., 1]**2
        
        sb = torch.exp(-0.5 * rs2 / self.std_rad**2)*self.I
        self.surface_brightness = sb
        return sb