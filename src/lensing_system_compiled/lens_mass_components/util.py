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
    if (q < 0.8).any():
        print("Warning: some q < 0.8 in this _hyp2f1_series implementation may not converge")
    
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

