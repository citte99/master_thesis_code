import torch

def _hyp2f1_series(z, r2, t, q, max_terms=15):
    """
    Batched implementation of the hypergeometric 2F1 series for the PEMD lens model.
    
    Parameters:
      z       : Complex tensor of shape [B, ...] representing the (complex) coordinate.
      r2      : Tensor of shape [B, ...] representing the squared elliptical radius.
      t       : Tensor of shape [B, ...] representing the power-law slope.
      q       : Tensor of shape [B, ...] representing the axis ratio.
      max_terms: Maximum number of terms in the series expansion.
      
    Returns:
      f       : Complex tensor of the same shape as z, containing the series evaluation.
    
    Note:
      A warning is printed if any element of q is less than 0.8, since convergence
      issues may occur in that regime.
    """

    raise NotImplementedError("This function implementation does not converge. The angular implemetation is used instead.")
    #if (q < 0.8).any():
    #    print("Warning: some q < 0.8 in this _hyp2f1_series implementation may not converge")
    
    # Compute qp = (1 - q^2) / q^2, with q being batched.
    qp = (1 - q**2) / q**2
    # Compute w2 = qp * r2 / z^2. Division is elementwise.
    w2 = qp * r2 / (z**2)
    
    # Compute u = 0.5 * (1 - sqrt(1 - w2))
    u = 0.5 * (1.0 - torch.sqrt(1.0 - w2))
    
    # Initialize u_n and a_n as tensors of ones with the same shape (and type) as u.
    u_n = torch.ones_like(u)  # u_n will accumulate powers of u
    a_n = torch.ones_like(u)  # a_n accumulates the coefficient product
    # Initialize the series sum.
    f = a_n * u_n
    
    # Sum the series for max_terms iterations.
    for n in range(max_terms):
        u_n = u_n * u  # Increase power: u^(n+1)
        # Compute the multiplier factor elementwise.
        # Note: the operations are broadcasted against t.
        num = (2 * n + 4) - 2 * t
        den = (2 * n + 4) - t
        a_n = a_n * (num / den)
        f = f + a_n * u_n
        
    return f


# # Wrap it in torch.compile
# compiled_hyp2f1 = torch.compile(_hyp2f1_series, backend="inductor")

# # Use exactly like the Python version:
# z = torch.randn(32, 100, 100, dtype=torch.cfloat, device="cuda")
# r2 = torch.rand(32, 100, 100, device="cuda")
# t  = torch.rand(32, 100, 100, device="cuda")
# q  = torch.rand(32, 100, 100, device="cuda") * 0.5 + 0.5

# # Warm-up (compilation)
# _ = compiled_hyp2f1(z, r2, t, q)


# ================================================================
# 1. Angular-only hypergeometric series  (always |arg|<1)
# ================================================================
from torch import Tensor

# -------------------------------------------------------------------
# helper: angular Gauss series  2F1(1, t/2 ; 2−t/2 ; z_ang)
# -------------------------------------------------------------------
def _hyp2f1_angular_series(z_ang: torch.Tensor,
                           t: torch.Tensor,
                           max_terms: int = 40) -> torch.Tensor:
    a = 1.0                       # constant
    b = 0.5 * t                   # t/2  (broadcasts)
    c = 2.0 - 0.5 * t             # 2 − t/2

    # promote to same dtype/device as z_ang once
    a = torch.as_tensor(a, dtype=z_ang.dtype, device=z_ang.device)
    b = b.to(dtype=z_ang.dtype, device=z_ang.device)
    c = c.to(dtype=z_ang.dtype, device=z_ang.device)

    term = torch.ones_like(z_ang)        # n = 0
    F    = term.clone()

    for n in range(1, max_terms):
        term = term * ((a+n-1)*(b+n-1) / ((c+n-1)*n)) * z_ang
        F   += term
    return F




from typing import Optional

# TorchScript-friendly version of the angular hypergeometric series





import torch
from torch import Tensor

@torch.jit.script
def _hyp2f1_angular_series_compiled(
    z_ang: Tensor,
    t: Tensor,
    max_terms: int = 15,
    a: float = 1.0,          # <-- Python float default
    c_base: float = 2.0      # <-- Python float default
) -> Tensor:
    b = t * 0.5
    term = torch.ones_like(z_ang)
    F = term
    for n in range(1, max_terms):
        # an is a float, bn/cn are Tensors
        an = a + (n - 1)
        bn = b + (n - 1)
        # subtract Python float from Tensor b to get a Tensor c
        cn = (c_base - b) + (n - 1)
        coeff = (an * bn) / (cn * n)
        term = term * coeff * z_ang
        F = F + term
    return F

