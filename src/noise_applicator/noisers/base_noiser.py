from abc import ABC, abstractmethod
from noise_applicator.registry import NOISERS_REGISTRY, INSTRUMENTS_REGISTRY
from config import PSFS_DIR
import torch
from typing import Union, Any
from dataclasses import dataclass
import math

class BaseNoiser(ABC):
    """
        A noiser takes as input a tensor image or a tensor batch of images, are applies noise.
        Some arguments will be allowed, for example to choose a particular PSF.

    """

    @abstractmethod
    def __call__(self, image_s: torch.Tensor)-> torch.Tensor:
        """
            This method has to be implemented by every child class.
            But a base logic of checking the correct form of image_s is implemented here, and must be called with super.
            For ease of implementation of the child classes, single images shape will be augmented to [batch, channels, N_x, N_y]

        """

        return _to_batch_shape(image_s)

    def set_device(self, device):
        self.device = device


def _to_batch_shape(image_s):
    
        if image_s.dim() == 2:
            image_s= image_s[None, None, ...]

        elif image_s.dim() == 3:
             "here we should know if the dimension already there is channel or batch."
             image_s= image_s[None, ...]
        elif image_s.dim() > 4:
             raise ValueError(f"The dimension of the provided images, shape {image_s.shape} is bigger than 4.")
        
        return image_s



@NOISERS_REGISTRY.register()
class GaussNoiser(BaseNoiser):
    def __init__(self, sigma: Union[float, torch.Tensor]):
        
         self.sigma= torch.as_tensor(sigma)
     
    def __call__(self, image_s: torch.Tensor)-> torch.Tensor:
        images= super().__call__(image_s)
        B, C, H, W = images.shape

        # make gauss noise matching shape of image_s
        noise = torch.randn( B, C, H, W)*self.sigma
        noisy_images = images + noise
        return noisy_images



"""
    Instrument response:

    -convolve with psf,
    -add uniform sky brighness (M_vis is given, how to convert to surface brighness?)
    -expected counts from
        -zero point (given in magnitudes)
        -exposure time

    Starting from I in W/m^2/sr,
    the photon counts per square meter per second of each pixel are given by
    B = I * pixel_arcsec_sa/ photon energy
    B * A is then the photon count per per second, that we multiply by the exposure time to get
    P = T * A * B the total expected photon count per pixel, if there was no loss of photons
    

    which we need to multiply by the troughput and quantum efficiency to get the real expected photon counts
"""
class PSF(ABC):
    @abstractmethod
    def get_tensor_psf(pixel_scale):
        pass
    pass



class AnalyticPSF():
     #pixel_scale, pixel_size
    def get_tensor_psf(pixel_scale):
        pass
    pass


class GaussPSF(AnalyticPSF):
    def __init__(self, FWHM_arcsec: float, n_sigma_trunc: float = 4.0):
        self.FWHM_arcsec    = FWHM_arcsec
        self.sigma_arcsec   = FWHM_arcsec / (2 * math.sqrt(2 * math.log(2)))
        self.n_sigma_trunc  = n_sigma_trunc

    def get_tensor_psf(self, pixel_scale: float) -> torch.Tensor:
        # how far out (in arcsec) to truncate the Gaussian
        radius_arcsec = self.n_sigma_trunc * self.sigma_arcsec

        # 2) convert to "half‐width in pixels", rounding to nearest int
        half_pix = int(radius_arcsec / pixel_scale + 0.5)
        npix     = 2 * half_pix + 1
        coords = torch.arange(npix, dtype=torch.float32) - half_pix  # [..., -half_pix, ... , +half_pix]
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")       # shape (npix, npix)
        xx = xx * pixel_scale
        yy = yy * pixel_scale
        rsq = xx**2 + yy**2
        psf  = torch.exp(-0.5 * rsq / (self.sigma_arcsec**2))
        psf = psf / psf.sum()
        return psf.unsqueeze(0).unsqueeze(0)  # shape (1,1,npix,npix)
    
    def get_max(self, pixel_scale: float) -> float: # this method is ad hoc for this psf
        """
        Returns the maximum (peak) value of the normalized PSF.
        """
        psf_tensor = self.get_tensor_psf(pixel_scale)
        return psf_tensor.max().item()


class PthPSF(PSF): 
    def __init__(self, psf_name):
        path = PSFS_DIR / "processed_psfs" / (psf_name + ".pth")
        psf_tensor = torch.as_tensor(torch.load(path, weights_only=False))
        self.psf_tensor = psf_tensor.unsqueeze(0).unsqueeze(0)

    def get_tensor_psf(self, pixel_scale=None):
        print("The PthPSF is loaded as it is. All the physical considerations must be done building it.")
        #here we will have to check that the pixel size matches
        #the fits file, or eventually make an interpolation but not very meaningful
        return self.psf_tensor
    



@dataclass
class Instrument:
    t_obs : torch.Tensor
    pixel_arcsec : torch.Tensor
    zero_point : torch.Tensor
    sky_mag : torch.Tensor
    gain : torch.Tensor
    eff_with_f: torch.Tensor
    psf: PSF


EuclidVis = Instrument(
    t_obs        = torch.tensor(1695), #s
    pixel_arcsec = torch.tensor(0.1),
    zero_point   = torch.tensor(25.2),
    sky_mag      = torch.tensor(22.2),
    gain         = torch.tensor(1.5),
    eff_with_f   = torch.tensor(2.12e14), # Hz
    psf          = GaussPSF(FWHM_arcsec=0.16, n_sigma_trunc=3)
)
INSTRUMENTS_REGISTRY.add_instance("EuclidVis",EuclidVis)





@NOISERS_REGISTRY.register()
class PoissonNoiser(BaseNoiser):
    """
        We need to convert an intensity I over the band into photon counts per pixel, and apply the shot 
        noise to them, then convert back to an intensity
    """
    def __init__(self, Instrument: Instrument):
        self.instrument = Instrument
        self.K = (
            10 ** (0.4 * (self.instrument.zero_point + 48.60))
            * self.instrument.pixel_arcsec**2
            * self.instrument.gain
        )


    def __call__(self, image_s: torch.Tensor)-> torch.Tensor:
        images = super().__call__(image_s)
        B, C, H, W = images.shape


        # converting surface brighness [erg s^-1 cm ^ -2 arcsec^ -2 ]
        # divide by effective f band width
        I_nu  = images / self.instrument.eff_with_f
        # to AB magnitude [mag / arcsec ^ -2]
        m_ab = -2.5*torch.log10(I_nu)-48.60



        # assuming the input is a surface brighness AB magnitude
        R = 10**(-0.4 * (m_ab - self.instrument.zero_point)) # ADU/s/ arcsec^2
        # convert ADU to electron count



        R_sky = 10**(-0.4 * (self.instrument.sky_mag - self.instrument.zero_point)) 
        R_pix = (R+ R_sky) * self.instrument.pixel_arcsec**2

        exp_phot      = R_pix * self.instrument.t_obs * self.instrument.gain
        neg_count = neg_count = (exp_phot < 0).sum().item()

        if neg_count:
            print(f"[WARN] Clamping {neg_count} negative λ’s to zero.")
            
        exp_phot = torch.clamp(exp_phot, min=0.0)

        pois_sample   = torch.poisson(exp_phot)
        phot_per_sec = pois_sample / self.instrument.t_obs
        # avoid log10(0)
        phot_per_sec = phot_per_sec.clamp(min=1e-10)
        #m_ab = -2.5 * torch.log10(Adu_per_sec) + self.instrument.zero_point
        
        # invert: I_nu = phot_per_sec / K
        I_nu_rec = phot_per_sec / self.K

        # back to surface brightness units:
        I_rec = I_nu_rec * self.instrument.eff_with_f
        
        return I_rec










import torch.nn.functional as F 

@NOISERS_REGISTRY.register()
class PSFConvolveNoiser(BaseNoiser):
    def __init__(self, psf : PSF, pixel_scale: float, device = "cuda"):
         self.PSF = psf
         self.tensor_PSF_filter = psf.get_tensor_psf(pixel_scale)
         self.tensor_PSF_filter=self.tensor_PSF_filter.to(device)

    def __call__(self, image_s: torch.Tensor)-> torch.Tensor:
        images = super().__call__(image_s)
        B, C, H, W = images.shape
        convolved = F.conv2d(images, self.tensor_PSF_filter, padding="same")
        return convolved


@NOISERS_REGISTRY.register()
class PSFConvolveFFTNoiser(BaseNoiser):
    """
        the assumption is that the input psf has already double the size of the image.
        This can be checked.
    """
    def __init__(self, psf : PSF, device = "cuda"):
        self.PSF = psf
        self.tensor_PSF_filter = psf.get_tensor_psf(pixel_scale)
        self.tensor_PSF_filter = self.tensor_PSF_filter.to(device)
        self._first_call = True

    def __call__(self, image_s: torch.Tensor) -> torch.Tensor:
        images = super().__call__(image_s)
        B, C, H, W = images.shape
        
        # Check PSF size on first call
        if self._first_call:
            psf_h, psf_w = self.tensor_PSF_filter.shape[-2:]
            if psf_h != 2*H or psf_w != 2*W:
                raise ValueError(f"PSF size {(psf_h, psf_w)} must be exactly 2x the image size {(H, W)}. "
                            f"Expected PSF size: {(2*H, 2*W)}")
            self._first_call = False
        
        # FFT of images (no padding needed since PSF is already 2x size)
        images_fft = torch.fft.fft2(images)
        # FFT of PSF (assuming it's already 2x the image size)
        psf_fft = torch.fft.fft2(self.tensor_PSF_filter, s=(H, W))
        # Multiply in frequency domain
        convolved_fft = images_fft * psf_fft
        # Inverse FFT and take real part
        convolved = torch.fft.ifft2(convolved_fft).real
        return convolved



@NOISERS_REGISTRY.register()
class EuclidNoiser(BaseNoiser):
    def __init__(self, device= 'cuda'):
        self.conv_noiser=PSFConvolveNoiser(
            psf=EuclidVis.psf, pixel_scale=EuclidVis.pixel_arcsec, device=device
        )
        self.poisson_noiser = PoissonNoiser(EuclidVis)
    
    def set_device(self, device):
        self.device = device
        self.conv_noiser=PSFConvolveNoiser(
            psf=EuclidVis.psf, pixel_scale=EuclidVis.pixel_arcsec, device=self.device
        )

    def __call__(self, image_s: torch.Tensor)-> torch.Tensor:
        images = super().__call__(image_s)
        B, C, H, W = images.shape
        
        conv_images = self.conv_noiser(images)
        poiss_images = self.poisson_noiser(conv_images)

        return (poiss_images)


@NOISERS_REGISTRY.register()
class EuclidNoiserInterfPSF(BaseNoiser):
    def __init__(self, device= 'cuda'):
        self.psf = PthPSF(psf_name="temp_bad_psf")
        self.conv_noiser=PSFConvolveFFTNoiser(
            psf = self.psf, device=device
        )
        self.poisson_noiser = PoissonNoiser(EuclidVis)
    
    def set_device(self, device):
        self.device = device
        self.conv_noiser=PSFConvolveFFTNoiser(
            psf = self.psf, device=device
        )

    def __call__(self, image_s: torch.Tensor)-> torch.Tensor:
        images = super().__call__(image_s)
        B, C, H, W = images.shape
        means = images.mean(dim=(2, 3), keepdim=True)

        conv_images = self.conv_noiser(images)
        #clip the images to 0 if they are negative
        conv_images = torch.clamp(conv_images, min=0.0)

        conv_means = conv_images.mean(dim=(2, 3), keepdim=True)
        #rescale the mean to the original image mean
        conv_images = conv_images * (means / conv_means)

        

        poiss_images = self.poisson_noiser(conv_images)

        return (poiss_images)

class AlmaNoiser(BaseNoiser):
     pass



if __name__ == "__main__":
    from skimage import data
    import numpy as np
    import matplotlib.pyplot as plt

    # img = data.camera()
    # img = -(img/img.max()*1)+20
    # m0=20
    
    # lets make a fake image, having appropriate ab magnitude around 22-24
    # for euclid a fast chat gpt search gives ~ 28 ab per arcsec

    # lets make a ring , 
    pixel_scale = 0.1 #arcsec,
    FOV_arcsec = 8 #arcsec
    npix = int(FOV_arcsec / pixel_scale)
    coords = np.linspace (-4., 4., npix)
    xx, yy = np.meshgrid(coords, coords, indexing='xy')
    R = xx**2 + yy**2

    # m pix+ = mu - 2.5 log10 A_pix
    #referece_mag_bright_pix = 28-2.5*np.log10(pixel_scale**2)
    I_0 = 1e-14  # Peak intensity in [erg s^-1 cm^-2 arcsec^-2]
    R_0 = 2.0    # Radius of the peak in arcseconds
    sigma = 0.1  # Width of the Gaussian in arcseconds

    SB = I_0 * np.exp(-((np.sqrt(R) - R_0)**2) / (2 * sigma**2))  
    print(SB.max()) #gives 10-14, which makes sense with the I vis expected

    img= SB



    plt.imshow(img)
    plt.colorbar()
    plt.show()
    
    print("Gaussian")
    Noiser= GaussNoiser(sigma= torch.tensor(0.1))
    img_tensor = torch.tensor (img).float()
    noisy_image = Noiser(img_tensor)

    plt.imshow(noisy_image[0][0].cpu())
    plt.show()

    img_tensor = img_tensor.to("cuda")
    print("Poisson")
    Noiser = PoissonNoiser(Instrument= EuclidVis)
    noisy_image = Noiser(img_tensor)
    plt.imshow(noisy_image[0][0].cpu())
    plt.show()

    print("Psf gauss")
    psf= GaussPSF(FWHM_arcsec=0.03, n_sigma_trunc=3)
    Noiser = PSFConvolveNoiser(psf, 0.01)
    noisy_image = Noiser(img_tensor)
    plt.imshow(noisy_image[0][0].cpu())
    plt.show()

    
    print("Euclid")
    Noiser = EuclidNoiser()
    noisy_image = Noiser(img_tensor)
    plt.imshow(noisy_image[0][0].cpu())
    plt.show()


    print("FFT + euclid poisson")
    Noiser = EuclidNoiserInterfPSF()
    noisy_image = Noiser(img_tensor)
    plt.imshow(noisy_image[0][0].cpu())
    plt.show()