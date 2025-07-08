import torch

def _translate_rotate(x, xc, th_rad):
    #a single vector applied to all points and a single rotation angle!
    return torch.view_as_real(torch.exp(-1j*th_rad)*torch.view_as_complex(x-xc))
