

from catalog_manager.make_system_dicts_new import Sampler, SamplingInputs,  RedshiftsVelDispModes, MainLensModes, SecondaryLensModes, SubStrucModes, SourceModes, SourcePosModes
import numpy as np
from config import CATALOGS_DIR
from shared_utils.physics_relations import r_max_moline

min_mass_log = 10.999
sampling_inputs = SamplingInputs(

        ThetaE_min   = 0.5,
        ThetaE_max   = 2.5,
        z_min=0.0,
        z_max=5.0,
        prior_lens_ThetaE=[0.5, 2.5],

        prior_lens_orient=[0, 2.* np.pi],
        prior_lens_pos=[-0.5, 0.5],
        prior_lens_q= [0.2, 1.0],
        prior_lens_slope_normal=[1.0, 0.1],
        prior_lens_VelDisp = [50, 400], 
    
        prior_sub_max_n = 1,
        prior_sub_pos= [- 3.0, + 3.0 ], # according to conor
        prior_sub_log_mass= [min_mass_log, 11.0], 
        prior_sub_r_max = [r_max_moline(10**min_mass_log) , r_max_moline(10**11.0) ],

        prior_source_I = 5e-14,         #if only one number, is fixed
        prior_source_std_main = 1.0, #kpc, a little bit low
        prior_source_frac_of_theta_pos= 0.3, #similar to conor
        prior_source_orient= 0., # not elliptical
        prior_source_q = 1. , # not elliptical

        prior_shear_d = [0., np.pi],
        prior_shear_s = [0., 0.1]
    )



My_Pypeline = [
    RedshiftsVelDispModes.RESAMPLE_THETA,
    MainLensModes.PEMD,
    SecondaryLensModes.EXTERNAL_SHEAR,
    SubStrucModes.NFW_SUBS_FREE_R_MAX,
    SourceModes.GAUSS_SOURCE,
    SourcePosModes.RAND_FRAC_THETA_E

]

   

full_sys_conf = Sampler(
    My_Pypeline,
    sampling_inputs,
    N_samples=100000,
    cat_name="min_mass_10e11_test"
)

full_sys_conf = Sampler(
    My_Pypeline,
    sampling_inputs,
    N_samples=5_000_000,
    cat_name="min_mass_10e11_long"
)


min_mass_log = 10.0 
sampling_inputs = SamplingInputs(

        ThetaE_min   = 0.5,
        ThetaE_max   = 2.5,
        z_min=0.0,
        z_max=5.0,
        prior_lens_ThetaE=[0.5, 2.5],

        prior_lens_orient=[0, 2.* np.pi],
        prior_lens_pos=[-0.5, 0.5],
        prior_lens_q= [0.2, 1.0],
        prior_lens_slope_normal=[1.0, 0.1],
        prior_lens_VelDisp = [50, 400], 
    
        prior_sub_max_n = 1,
        prior_sub_pos= [- 3.0, + 3.0 ], # according to conor
        prior_sub_log_mass= [min_mass_log, 11.0], 
        prior_sub_r_max = [r_max_moline(10**min_mass_log) , r_max_moline(10**11.0) ],

        prior_source_I = 5e-14,         #if only one number, is fixed
        prior_source_std_main = 1.0, #kpc, a little bit low
        prior_source_frac_of_theta_pos= 0.3, #similar to conor
        prior_source_orient= 0., # not elliptical
        prior_source_q = 1. , # not elliptical

        prior_shear_d = [0., np.pi],
        prior_shear_s = [0., 0.1]
    )



My_Pypeline = [
    RedshiftsVelDispModes.RESAMPLE_THETA,
    MainLensModes.PEMD,
    SecondaryLensModes.EXTERNAL_SHEAR,
    SubStrucModes.NFW_SUBS_FREE_R_MAX,
    SourceModes.GAUSS_SOURCE,
    SourcePosModes.RAND_FRAC_THETA_E

]

   

full_sys_conf = Sampler(
    My_Pypeline,
    sampling_inputs,
    N_samples=100000,
    cat_name="min_mass_10e10_test"
)

full_sys_conf = Sampler(
    My_Pypeline,
    sampling_inputs,
    N_samples=5_000_000,
    cat_name="min_mass_10e10_long"
)




min_mass_log = 9.0
sampling_inputs = SamplingInputs(

        ThetaE_min   = 0.5,
        ThetaE_max   = 2.5,
        z_min=0.0,
        z_max=5.0,
        prior_lens_ThetaE=[0.5, 2.5],

        prior_lens_orient=[0, 2.* np.pi],
        prior_lens_pos=[-0.5, 0.5],
        prior_lens_q= [0.2, 1.0],
        prior_lens_slope_normal=[1.0, 0.1],
        prior_lens_VelDisp = [50, 400], 
    
        prior_sub_max_n = 1,
        prior_sub_pos= [- 3.0, + 3.0 ], # according to conor
        prior_sub_log_mass= [min_mass_log, 11.0], 
        prior_sub_r_max = [r_max_moline(10**min_mass_log) , r_max_moline(10**11.0) ],

        prior_source_I = 5e-14,         #if only one number, is fixed
        prior_source_std_main = 1.0, #kpc, a little bit low
        prior_source_frac_of_theta_pos= 0.3, #similar to conor
        prior_source_orient= 0., # not elliptical
        prior_source_q = 1. , # not elliptical

        prior_shear_d = [0., np.pi],
        prior_shear_s = [0., 0.1]
    )



My_Pypeline = [
    RedshiftsVelDispModes.RESAMPLE_THETA,
    MainLensModes.PEMD,
    SecondaryLensModes.EXTERNAL_SHEAR,
    SubStrucModes.NFW_SUBS_FREE_R_MAX,
    SourceModes.GAUSS_SOURCE,
    SourcePosModes.RAND_FRAC_THETA_E

]

   

full_sys_conf = Sampler(
    My_Pypeline,
    sampling_inputs,
    N_samples=100000,
    cat_name="min_mass_10e9_test"
)

full_sys_conf = Sampler(
    My_Pypeline,
    sampling_inputs,
    N_samples=5_000_000,
    cat_name="min_mass_10e9_long"
)



min_mass_log = 8.6
sampling_inputs = SamplingInputs(

        ThetaE_min   = 0.5,
        ThetaE_max   = 2.5,
        z_min=0.0,
        z_max=5.0,
        prior_lens_ThetaE=[0.5, 2.5],

        prior_lens_orient=[0, 2.* np.pi],
        prior_lens_pos=[-0.5, 0.5],
        prior_lens_q= [0.2, 1.0],
        prior_lens_slope_normal=[1.0, 0.1],
        prior_lens_VelDisp = [50, 400], 
    
        prior_sub_max_n = 1,
        prior_sub_pos= [- 3.0, + 3.0 ], # according to conor
        prior_sub_log_mass= [min_mass_log, 11.0], 
        prior_sub_r_max = [r_max_moline(10**min_mass_log) , r_max_moline(10**11.0) ],

        prior_source_I = 5e-14,         #if only one number, is fixed
        prior_source_std_main = 1.0, #kpc, a little bit low
        prior_source_frac_of_theta_pos= 0.3, #similar to conor
        prior_source_orient= 0., # not elliptical
        prior_source_q = 1. , # not elliptical

        prior_shear_d = [0., np.pi],
        prior_shear_s = [0., 0.1]
    )



My_Pypeline = [
    RedshiftsVelDispModes.RESAMPLE_THETA,
    MainLensModes.PEMD,
    SecondaryLensModes.EXTERNAL_SHEAR,
    SubStrucModes.NFW_SUBS_FREE_R_MAX,
    SourceModes.GAUSS_SOURCE,
    SourcePosModes.RAND_FRAC_THETA_E

]

   

full_sys_conf = Sampler(
    My_Pypeline,
    sampling_inputs,
    N_samples=100000,
    cat_name="min_mass_10e8_6_test"
)

full_sys_conf = Sampler(
    My_Pypeline,
    sampling_inputs,
    N_samples=5_000_000,
    cat_name="min_mass_10e8_6_long"
)




import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.optimize import curve_fit

def plot_tensor(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0).squeeze(0)
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    plt.imshow(tensor.cpu())
    plt.colorbar()
    plt.show()

def gaussian_2d(coords, A, x0, y0, σx, σy, offset):
    """
    coords: tuple of (X, Y) meshgrids, each of shape (H, W)
    A: amplitude
    x0, y0: center
    σx, σy: standard deviations
    offset: constant background
    """
    X, Y = coords
    exponent = ((X - x0)**2)/(2*σx**2) + ((Y - y0)**2)/(2*σy**2)
    return offset + A * np.exp(-exponent)

def fit_gaussian_and_get_sigma(tensor_2d):
    """
    Fits a 2D Gaussian to the input tensor and returns the sigma values in pixels.

    Parameters:
    tensor_2d (torch.Tensor): A 2D tensor representing the data.

    Returns:
    tuple: σx and σy in pixels.
    """
    if isinstance(tensor_2d, torch.Tensor):
        data = tensor_2d.detach().cpu().numpy()
    else:
        data = tensor_2d

    H, W = data.shape
    Y, X = np.indices((H, W))

    total = data.sum()
    x0 = (X * data).sum() / total
    y0 = (Y * data).sum() / total

    # Rough σ from marginal variances
    σx = np.sqrt(np.abs(((X - x0)**2 * data).sum() / total))
    σy = np.sqrt(np.abs(((Y - y0)**2 * data).sum() / total))

    A = data.max() - data.min()
    offset = data.min()

    initial_guess = (A, x0, y0, σx, σy, offset)

    lb = [-np.inf, -np.inf, -np.inf, 0.0, 0.0, -np.inf]
    ub = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    # Flatten everything
    coords = np.vstack([X.ravel(), Y.ravel()])
    values = data.ravel()

    popt, _ = curve_fit(
        gaussian_2d,
        coords,
        values,
        p0=initial_guess,
        bounds=(lb, ub)
    )

    _, _, _, σx_fit, σy_fit, _ = popt

    return σx_fit, σy_fit



import os

from config import PSFS_DIR
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

psf_name="devon_first_advice_psf_5_pix_16_arcsec"

psf_path = os.path.join(PSFS_DIR, psf_name + '.fits')

with fits.open(psf_path) as hdul:
                    psf_data = hdul[0].data
                    psf_data = psf_data.byteswap().view(psf_data.dtype.newbyteorder())
            
            
psf_tensor = torch.from_numpy(psf_data).float().to("cuda")
print(f"SHAPE OF PSF = {psf_tensor.shape}")

plot_tensor(psf_tensor)

crop_border = 1890
# crop to get a good image where to fit the gaussian and get sigma.
cropped_tensor = psf_tensor[:, :, crop_border:-crop_border, crop_border: - crop_border]
plot_tensor(cropped_tensor)


# get the FWHM in pixels

data=cropped_tensor.cpu().numpy()[0][0]
H, W = data.shape
Y, X = np.indices((H, W))  - H//2
sigma_pix_x, sigma_pix_y = fit_gaussian_and_get_sigma(data)
print(f"Sigma x pix = {sigma_pix_x}, Sigma y pix = {sigma_pix_y}")
sigma = np.sqrt(sigma_pix_x * sigma_pix_y ) # geometrical mean
print(f"Sigma = {sigma}")
measured_fwhm = 2.* np.sqrt(2* np.log(2.)* sigma)


#Now lets build the psf tensor ready to use:
requested_fwhm_pix = 1.6 
requested_size = 160 # must be two times the image size

# Strategy: 

factor = measured_fwhm / requested_fwhm_pix
X_new, Y_new = (np.indices((requested_size, requested_size))-requested_size//2) * factor

# and now interpolate the values on the new grid 

# finally renormalize to 1 for a physical psf.
psf_data = psf_tensor[0][0].cpu().numpy() if hasattr(psf_tensor[0][0], 'cpu') else psf_tensor[0][0]

H, W = psf_data.shape

# Create 1D coordinate arrays (not 2D grids!)
y_coords = np.arange(H) - H//2  # Center coordinates
x_coords = np.arange(W) - W//2

# RegularGridInterpolator expects (grid_y, grid_x) order to match array indexing
interp = RegularGridInterpolator(
    (y_coords, x_coords),  # 1D coordinate arrays
    psf_data,              # 2D data array
    method= 'cubic',
    bounds_error=False,
    fill_value=0.0
)

new_data = interp((X_new, Y_new))


#watch out to the normalization!! here we scale the max of the main lobe to the max of the gaussian
from noise_applicator.noisers.base_noiser import GaussPSF
psf = GaussPSF(FWHM_arcsec=0.16)
max_gauss_psf = psf.get_max(pixel_scale = 0.01)

print(max_gauss_psf)

new_psf_tensor = new_data / (new_data.max()) * max_gauss_psf

# THIS NORMALIZES ONLY FOR THE POSITIVE VALUES
# psf_positive_sum = new_data[new_data > 0].sum()
# new_psf_tensor = new_data / psf_positive_sum

plt.imshow(new_psf_tensor)
plt.colorbar()
plt.show()

print(f"psf sum should be 1 for gauss, here it is {new_psf_tensor.sum()}")

name_psf = "temp_bad_psf"
os.makedirs(PSFS_DIR / "processed_psfs", exist_ok=True)
torch.save(new_psf_tensor, PSFS_DIR /"processed_psfs" / (name_psf + ".pth")) 