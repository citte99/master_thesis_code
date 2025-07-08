import torch
import numpy as np
from astropy.cosmology import Planck18 as cosmo


def uniform_prior(n_samples, min_value, max_value):
    return np.random.uniform(min_value, max_value, n_samples)

def log_uniform_prior(n_samples, min_value, max_value):
    return np.exp(np.random.uniform(np.log(min_value), np.log(max_value), n_samples))



def sample_redshift_comoving_volume(n_samples, zmin=0.0, zmax=5.0):
    """
    Sample redshifts according to the comoving volume distribution 
    using the Planck18 cosmology.
    
    Parameters:
    - n_samples : int
        Number of redshifts to sample.
    - zmin : float, optional
        Minimum redshift (default is 0.0).
    - zmax : float, optional
        Maximum redshift (default is 5.0).
    
    Returns:
    - sampled_z : numpy.ndarray
        Array of redshifts sampled according to the comoving volume.
    """
    # Create a fine redshift grid between zmin and zmax.
    z_grid = np.linspace(zmin, zmax, 10000)
    
    # Compute the comoving volume for each redshift in the grid.
    # The .value extracts the numerical value in the default unit (e.g., Mpc^3).
    vol_grid = cosmo.comoving_volume(z_grid).value
    
    # Get the minimum and maximum comoving volumes in the range.
    vol_min = cosmo.comoving_volume(zmin).value
    vol_max = cosmo.comoving_volume(zmax).value
    
    # Generate uniform random samples in the comoving volume range.
    random_volumes = np.random.uniform(vol_min, vol_max, n_samples)
    
    # Invert the cumulative distribution by interpolating the redshift grid.
    sampled_z = np.interp(random_volumes, vol_grid, z_grid)
    
    return sampled_z


def random_pos_in_circle(radius: float, n_samples: int, rng= None):
    """
    Draw `n_samples` points uniformly from the interior of a circle.

    Parameters
    ----------
    radius : float
        Radius of the circle (centered at the origin).
    n_samples : int
        Number of random points to generate.
    rng : np.random.Generator | None
        Optional NumPy random generator for reproducible results.

    Returns
    -------
    np.ndarray
        Shape (n_samples, 2); each row is an (x, y) pair.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Random angles ∈ [0, 2π)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)

    # 2. Radial distances scaled by √U to keep area density uniform
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_samples))

    # 3. Convert polar → Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.column_stack((x, y))





import numpy as np
from numba import njit, prange
from astropy.cosmology import Planck18 as cosmo

# —————————————————————————————————————————————————————————————
# 1) One-time precompute (in pure Python, NOT inside @njit):
zmin, zmax = 0.0, 5.0
Nz         = 10_000
z_grid     = np.linspace(zmin, zmax, Nz)
# comoving volume V(z) and comoving distance χ(z):
vol_grid   = cosmo.comoving_volume(z_grid).value
chi_grid   = cosmo.comoving_distance(z_grid).value
# —————————————————————————————————————————————————————————————

@njit
def _searchsorted(a, v):
    lo, hi = 0, a.shape[0]-1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] <= v:
            lo = mid + 1
        else:
            hi = mid - 1
    # hi such that a[hi] <= v < a[hi+1]
    if hi < 0:
        return 0
    if hi >= a.shape[0]-1:
        return a.shape[0]-2
    return hi

@njit
def sample_redshift_comoving_volume_jit(n_samples, z_grid, vol_grid):
    n_grid = z_grid.shape[0]
    vol_min = vol_grid[0]
    vol_max = vol_grid[n_grid - 1]
    out     = np.empty(n_samples, dtype=z_grid.dtype)

    for i in range(n_samples):
        u  = np.random.random()
        rv = vol_min + u * (vol_max - vol_min)
        idx = _searchsorted(vol_grid, rv)
        dv  = vol_grid[idx+1] - vol_grid[idx]
        t   = (dv > 0.0) and ((rv - vol_grid[idx]) / dv) or 0.0
        dz  = z_grid[idx+1] - z_grid[idx]
        out[i] = z_grid[idx] + t * dz
    return out

@njit
def _interp1d(x, xp, yp):
    idx    = _searchsorted(xp, x)
    x0, x1 = xp[idx],   xp[idx+1]
    y0, y1 = yp[idx],   yp[idx+1]
    t      = (x1 > x0) and ((x - x0) / (x1 - x0)) or 0.0
    return y0 + t * (y1 - y0)



@njit(parallel=True)
def resample_theta(num_samples, oversampling_factor,
           min_einstein_angle, c,
           z_grid, vol_grid, chi_grid):
    # hardcode 4 arcsec in radians
    max_einstein_angle = 3.0 * np.pi / (180.0 * 3600.0)

    # 1) Oversample redshifts
    N      = int(num_samples * oversampling_factor)
    z_lens = sample_redshift_comoving_volume_jit(N, z_grid, vol_grid)
    z_src  = sample_redshift_comoving_volume_jit(N, z_grid, vol_grid)

    # 2) lens < source
    mask = z_lens < z_src
    M    = mask.sum()
    if M <= num_samples:
        raise ValueError("Too low oversampling rate at redshift ordering")
    z_l = z_lens[mask]
    z_s = z_src [mask]

    # 3) velocity dispersions
    vel = np.random.uniform(50.0, 400.0, M)

    # 4) compute D_l, D_s, D_ls
    D_l  = np.empty(M)
    D_s  = np.empty(M)
    D_ls = np.empty(M)
    for i in prange(M):
        chi_l    = _interp1d(z_l[i], z_grid, chi_grid)
        chi_s    = _interp1d(z_s[i], z_grid, chi_grid)
        D_l [i]  = chi_l / (1.0 + z_l[i])
        D_s [i]  = chi_s / (1.0 + z_s[i])
        D_ls[i]  = (chi_s - chi_l) / (1.0 + z_s[i])

    # 5) Einstein angles
    theta_E = 4.0 * np.pi * (vel / c)**2 * D_ls / D_s

    # 6) global θ_E cuts
    keep2 = (theta_E >= min_einstein_angle) & (theta_E <= max_einstein_angle)
    K     = keep2.sum()
    if K <= num_samples:
        raise ValueError("Too low oversampling rate at θ_E masking")
    z_l      = z_l     [keep2]
    z_s      = z_s     [keep2]
    vel      = vel     [keep2]
    D_l      = D_l     [keep2]
    D_s      = D_s     [keep2]
    D_ls     = D_ls    [keep2]
    theta    = theta_E [keep2]

    # 7) area-weighting ∝ θ^2
    theta_max = theta.max()
    p_keep    = (theta * theta) / (theta_max * theta_max)
    r         = np.random.random(p_keep.shape)
    keep3     = r < p_keep
    L         = keep3.sum()
    if L < num_samples:
        raise ValueError("Too low oversampling rate after weighting")
    
    #Finally, resample theta and get vel disp as a function of it
    
    z_l=z_l  [keep3][:num_samples]
    z_s=z_s  [keep3][:num_samples]
    D_l=D_l  [keep3][:num_samples]
    D_s=D_s  [keep3][:num_samples]
    D_ls=D_ls [keep3][:num_samples]
    
    theta =np.random.uniform(min_einstein_angle, max_einstein_angle, num_samples)
    vel  = np.sqrt(theta*c**2*D_s/(D_ls*4*np.pi))
    
    #theta[keep3][:num_samples]
    #vel  [keep3][:num_samples]
    
    
    
    
    # 8) final selection
    return (
        z_l,  
        z_s,  
        vel,  
        D_l,  
        D_s,  
        D_ls, 
        theta
    )

# Usage example:
# results = helper(
#     num_samples, oversampling_factor,
#     min_einstein_angle, c_value,
#     z_grid, vol_grid, chi_grid
# )
