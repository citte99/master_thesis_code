import torch
import torch.nn.functional as F
from torch.distributions import Uniform


class NoiseApplicator:
    def __init__(self, apply_poisson=False, sky_level=0., gain=None, gain_interval=None, psf=None, device=None):
        """
        
        """
        
        self.apply_poisson = apply_poisson
        self.sky_level = sky_level
        self.gain = gain
        self.psf = psf
        self.device = device 
        self.gain_interval=gain_interval

    
    def _add_sky_level(self, image):
        """Add sky background level to the image"""
        with torch.no_grad():
            # Add the sky level to the image
            image_with_sky = image + self.sky_level
        return image_with_sky
    
    def _convolve_psf(self, image):
        """Apply PSF convolution to the image"""
        # Apply convolution
        with torch.no_grad():
            convolved = F.conv2d(image, self.psf, padding='same')
        return convolved
    
    def _add_poisson_noise(self, image):

        if self.gain is None and self.gain_interval is None:
            raise ValueError("Gain must be provided when applying Poisson noise")
            
        elif self.gain_interval is not None:
            low=torch.tensor(self.gain_interval[0])
            high=torch.tensor(self.gain_interval[1])
            N = image.shape[0]
            
            log_low, log_high = torch.log(low), torch.log(high)
            dist = Uniform(log_low, log_high)
            u = dist.sample((N,1,1,1)).to(image.device)
            self.gain = u.exp()
        
        with torch.no_grad():
            electrons = image * self.gain
            
            # Calculate noise in electrons
            noise_stddev_electrons = torch.sqrt(electrons)
    
            # Convert noise standard deviation back to ADU
            noise_stddev_adu = noise_stddev_electrons / self.gain
            
            # Generate Gaussian noise with stddev matching Poisson statistics
            noise = torch.randn_like(image) * noise_stddev_adu
            
            # Add the noise to the original image
            noisy_image = image + noise
            
        return noisy_image

    
    def apply(self, image):
        """
        Apply PSF convolution and noise to the input image based on the selected noise mode.
        
        Args:
            image (torch.Tensor): Input image of shape [batch, channels, height, width] or [channels, height, width]
            
        Returns:
            torch.Tensor: Processed image with convolution and noise
        """
        # Ensure image has batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Move image to the same device as PSF
        image = image.to(self.device)
        
        # Apply PSF convolution if PSF is provided
        if self.psf is not None:
            image = self._convolve_psf(image)

        # Add sky background level
        image = self._add_sky_level(image)

        # Apply Poisson noise if specified
        if self.apply_poisson:
            image = self._add_poisson_noise(image)
            
        return image
    