
import torch
import torch.nn.functional as F
from typing import List, Tuple
from ..base import CustomDatasetBase
from catalog_manager import CatalogManager
from shared_utils import recursive_to_tensor
from lensing_system import LensingSystem
from shared_utils import _grid_lens
from astropy.io import fits
from deep_learning.registry import DATASET_REGISTRY
from config import PSFS_DIR
import time
import os 

@DATASET_REGISTRY.register()
class AlmaSinglePsfDataset(CustomDatasetBase):
    """
    Memory-optimized dataset that produces images with noise.
    Built to work with both standard models and Vision Transformers.
    """
    def __init__(
        self, 
        catalog_name=None, 
        catalog_dict=None,
        samples_used="all", 
        image_data_type=torch.float32,
        psf_name=None, 
        noise_std=0.1, 
        threshold=None,
        broadcasting=False
    ):
        # Initialize the base class
        super().__init__(catalog_name, catalog_dict, samples_used, image_data_type)
        if broadcasting:
            raise ValueError("This dataset does not support broadcasting for the image generation yet.")
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device in AlmaSinglePsfDataset: {self.device}")
        
        # Store dataset-specific parameters
        self.noise_std = noise_std
        self.threshold = threshold
            
        # Convert catalog to tensors
        self.catalog = recursive_to_tensor(self.catalog, self.device)
            
        # Process original PSF
        if psf_name is not None:
            self.psf_name = psf_name
            try:
                psf_path = os.path.join(PSFS_DIR, self.psf_name + '.fits')
                with fits.open(psf_path) as hdul:
                    psf_data = hdul[0].data
                    psf_data = psf_data.byteswap().newbyteorder()
                
                # Convert to tensor
                psf_tensor = torch.from_numpy(psf_data).float().to(self.device)
                
                # Determine PSF dimensions and ensure it's a multiple of 4
                if psf_tensor.ndim == 2:
                    psf_size = psf_tensor.shape[0]
                elif psf_tensor.ndim == 3:
                    psf_size = psf_tensor.shape[1]
                elif psf_tensor.ndim == 4:
                    psf_size = psf_tensor.shape[2]
                else:
                    raise ValueError(f"Unexpected PSF dimensions: {psf_tensor.shape}")
                
                # Make PSF size a multiple of 4 if necessary
                adjusted_psf_size = (psf_size // 4) * 4
                if adjusted_psf_size != psf_size:
                    print(f"Adjusting PSF size from {psf_size} to {adjusted_psf_size} to ensure it's a multiple of 4")
                    # Resize PSF if needed
                    if psf_tensor.ndim == 2:
                        psf_tensor = F.interpolate(
                            psf_tensor.unsqueeze(0).unsqueeze(0), 
                            size=(adjusted_psf_size, adjusted_psf_size),
                            mode='bilinear'
                        ).squeeze(0).squeeze(0)
                    else:
                        # Handle other dimensions appropriately
                        psf_tensor = F.interpolate(
                            psf_tensor.unsqueeze(0) if psf_tensor.ndim == 3 else psf_tensor,
                            size=(adjusted_psf_size, adjusted_psf_size),
                            mode='bilinear'
                        )
                        if psf_tensor.ndim == 3:
                            psf_tensor = psf_tensor.squeeze(0)
                
                # Ensure PSF has correct dimensions for processing
                if psf_tensor.ndim == 2:
                    self.psf = psf_tensor.unsqueeze(0).unsqueeze(0)
                elif psf_tensor.ndim == 3:
                    self.psf = psf_tensor.unsqueeze(0)
                else:
                    self.psf = psf_tensor
                
                print(f"PSF side pixels: {self.psf.shape[-1]}")
                
            except Exception as e:
                raise ValueError(f"Error loading PSF: {e}")
            
            print(f"Original PSF shape: {self.psf.shape}")
            
            # Shift PSF for FFT - save GPU memory by pre-computing
            self.psf_shifted = torch.fft.ifftshift(self.psf, dim=(-2, -1)).to(self.device)
            self.psf = self.psf.to(self.device)
            
            # Create grid based on PSF size (half the size of PSF)
            grid_size = self.psf.shape[-1] // 2
            self.image_no_double_pixel_size=grid_size
            FOV = 8.0  # Field of view in arcseconds
            # Define a static variable to track the last print time
            if not hasattr(self, '_last_alert_time'):
                self._last_alert_time = 0

            print("alert: FOV is hardcoded to 8.0 arcseconds. To hide this, set alert=False")
            self.grid = _grid_lens(FOV, grid_size, device=self.device)
            print(f"Created grid with size {grid_size}x{grid_size} (half of PSF)")
        else:
            raise ValueError("PSF tensor must be provided for this dataset")
            
        print(f"Dataset initialized with noise_std={noise_std}, threshold={threshold}")


    def get_image_pixel_size(self):
        return self.image_no_double_pixel_size
        
    def _process_single_item(self, idx, grid):
        """Process a single dataset item"""
        # Get data from catalog manager
        system_dict_tensor = self.catalog["SL_systems"][idx]
        
        # Get label
        num_sub = system_dict_tensor["lens_model"]["num_substructures"]
        label_tensor = torch.tensor(int(num_sub > 0), device=self.device).long()
        
        # Generate image
        lensing_system = LensingSystem(system_dict_tensor, self.device)
        with torch.no_grad():
            # Get base image from lensing system
            image_tensor = lensing_system(grid)
            
            # Add channel dimension if needed
            if image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, label_tensor
        
    def _apply_psf_and_noise(self, images):
        """Apply PSF convolution and noise with memory optimization"""
        # Skip if nothing to apply
    
        batch_size, channels, H, W = images.shape
        
        # Apply PSF convolution
        if self.psf is not None:
            # Pad the images
            padding = (W//2, W//2, H//2, H//2)
            padded_images = F.pad(images, padding, mode='constant', value=0)
            
            # Compute FFT of padded images
            images_fft = torch.fft.fftn(padded_images, dim=(-2, -1))
            
            # Get appropriate slice of PSF_fft if dimensions don't match
            psf_fft = torch.fft.fftn(self.psf_shifted, dim=(-2, -1))
            
            if images_fft.shape[-2:] != psf_fft.shape[-2:]:
                print(f"Warning: FFT shape mismatch - images: {images_fft.shape}, PSF: {psf_fft.shape}")
                # Use direct noise addition instead
                processed_images = images + torch.randn_like(images) * self.noise_std
                return processed_images
            
            # Apply convolution in Fourier domain
            conv_fft = images_fft * psf_fft
            
            # Add Fourier domain noise if specified
            if self.noise_std > 0:
                # Add noise in Fourier domain (complex noise)
                noise_real = torch.randn_like(conv_fft.real) * self.noise_std
                noise_imag = torch.randn_like(conv_fft.imag) * self.noise_std
                noise_fourier = torch.complex(noise_real, noise_imag)
                conv_fft = conv_fft + noise_fourier
            
            # Transform back to spatial domain
            convolved_images = torch.fft.ifftn(conv_fft, dim=(-2, -1)).real
            
            # Crop back to original size
            start_h, start_w = H // 2, W // 2
            processed_images = convolved_images[:, :, start_h:start_h+H, start_w:start_w+W]
            
            # Clean up large intermediate tensors
            del padded_images, images_fft, psf_fft, conv_fft, convolved_images
            #the following seems to be slowing down.
            #torch.cuda.empty_cache()
            
        # If only noise is needed (no PSF)
        elif self.noise_std > 0:
            processed_images = images + torch.randn_like(images) * self.noise_std
        else:
            processed_images = images
        
        # Apply thresholding if specified
        if self.threshold is not None:
            processed_images = torch.where(processed_images > self.threshold, 
                                          processed_images, 
                                          torch.zeros_like(processed_images))
            
        return processed_images

    def get_batch(self, idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of samples indexed by idxs.
        
        Args:
            idxs: List of indices to retrieve
            
        Returns:
            Tuple of (images, labels) as tensors
        """
        # Use the pre-defined grid created during initialization
        grid = self.grid
        
        # Individual processing approach
        images = []
        labels = []
        
        for idx in idxs:
            image_tensor, label_tensor = self._process_single_item(idx, grid)
            images.append(image_tensor)
            labels.append(label_tensor)
        # Stack tensors into batches
        images_batch = torch.stack(images)
        
        # Apply PSF and noise to the whole batch at once
        if self.psf is not None or self.noise_std > 0:
            images_batch = self._apply_psf_and_noise(images_batch)
            
        labels_batch = torch.stack(labels)
        
        # put pixels in [0,1] if they aren't already
        images_batch = images_batch.float() / 255.0  

        # zero-centre & scale:  (x - 0.5) / 0.5
        images_batch = T.functional.normalize(images_batch, mean=[0.5], std=[0.5])
        
        return images_batch.to(dtype=self.image_data_type), labels_batch