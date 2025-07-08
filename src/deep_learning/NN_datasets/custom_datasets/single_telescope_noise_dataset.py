import torch
import torchvision.transforms as T
from typing import List, Tuple
from .no_noise_dataset import NoNoiseDataset
from catalog_manager import CatalogManager
from shared_utils import recursive_to_tensor, _grid_lens
from lensing_system import LensingSystem
from lensing_system_broadcasting import LensingSystem as LensingSystemBroadcasting
from deep_learning.registry import DATASET_REGISTRY
from noise_applicator import NoiseApplicator, GaussKernel

from .util import train_tf


@DATASET_REGISTRY.register()
class SingleTelescopeNoiseDataset(NoNoiseDataset):

    def __init__(self, catalog_name=None, catalog_dict=None, samples_used="all", image_data_type=torch.float32,
                upscaling=None,
                grid_width_arcsec=None,
                grid_pixel_side=None,
                final_transform=False,
                broadcasting=False,
                sky_level=0,
                kernel_size=None,
                kernel_sigma=None,
                gain=None,
                gain_interval=None, # this will be sampled uniformly in the log if provided
                device=None):
        

           
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if grid_width_arcsec is None or grid_pixel_side is None:    
            raise ValueError("grid_width_arcsec and grid_pixel_side must be provided")
        self.uncropped_grid = _grid_lens(grid_width_arcsec, grid_pixel_side, device=self.device)
        self.final_transform = final_transform
        self.broadcasting = broadcasting
        print(f"Using device: {self.device}")


        if kernel_size is not None and kernel_sigma is not None and (gain is not None or gain_interval is not None) and sky_level is not None:
            gauss_kernel = GaussKernel(kernel_size, sigma=kernel_sigma, device=self.device).get_kernel()
            self.psf_and_poisson_noise = NoiseApplicator(apply_poisson=True,
                                                    sky_level=sky_level,
                                                    gain=gain,
                                                    gain_interval=gain_interval, #this will be sampled uniformly in the log
                                                    psf=gauss_kernel,
                                                    device=self.device)
        else:
            raise ValueError("kernel_size, kernel_sigma,( gain of gain_interval) and sky_level must be provided for SingleTelescopeNoiseDataset")
            
        super().__init__(
            catalog_name, 
            catalog_dict, 
            samples_used, 
            image_data_type,
            grid_width_arcsec,
            grid_pixel_side,
            upscaling=upscaling,
            final_transform=False, # this has to be handles maybe after the noise application
            broadcasting=broadcasting,
            device=device
        ) 

    # def __len__(self): the len is defined in the base class

    def get_batch(self, idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of samples indexed by idxs.
        
        Args:
            idxs: List of indices to retrieve
            
        Returns:
            Tuple of (images, labels) as tensors
        """
        no_noise_image_batch, labels_batch= super().get_batch(idxs)
        images_batch = self.psf_and_poisson_noise.apply(no_noise_image_batch)
        
        #already unsqueeze the batch dimension
        #images_batch= images_batch.unsqueeze(1) 

        # Add random crop to the images: this is done in the main dataloader now. Not necessarly correct, since the noise is applied afterwards

        

        images_batch = self.unit_max(images_batch, use_log=False)   # or True
        
        return images_batch, labels_batch