import torch

def _translate_rotate(x, xc, th_rad):
    # Function remains unchanged as it already supports batched operations
    return torch.view_as_real(torch.exp(-1j*th_rad)*torch.view_as_complex(x-xc))