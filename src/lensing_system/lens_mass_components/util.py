import torch
import warnings

# make sure we only warn once per session
warnings.simplefilter("once", UserWarning)

def _hyp2f1_series(z, r2, t, q, max_terms=15):
    '''Hypergeometric 2F1 series for the PEMD lens model'''
    if q < 0.8:
        warnings.warn(
            "q<0.8 in this hyp2f1 implementation may not converge",
            UserWarning,
            stacklevel=2
        )

    qp = (1 - q**2) / q**2
    w2 = qp * r2 / z**2
    u = 0.5 * (1.0 - (1.0 - w2)**0.5)
    u_n = torch.ones_like(u)  # note u is complex
    a_n = 1.0
    f = a_n * u_n

    for n in range(max_terms):
        u_n = u_n * u
        a_n = a_n * ((2*n + 4 - 2*t)/(2*n + 4 - t))
        f = f + a_n * u_n

    return f

def _hyp2f1_angular_series(z_ang: torch.Tensor,
                           t: torch.Tensor,
                           max_terms: int = 15) -> torch.Tensor:
    """
    Plain Taylor expansion of 2F1(1, t/2; 2 - t/2; z_ang):
      a = 1, b = t/2, c = 2 - t/2
    """
    # make sure scalars broadcast with z_ang
    a = torch.as_tensor(1.0, dtype=z_ang.dtype, device=z_ang.device)
    b =     t / 2.0
    c = 2.0 - t / 2.0

    term = torch.ones_like(z_ang)  # n=0
    F    = term.clone()

    for n in range(1, max_terms):
        coeff = ( (a + n - 1) * (b + n - 1) ) / ( (c + n - 1) * n )
        term  = term * coeff * z_ang
        F     = F + term

    return F