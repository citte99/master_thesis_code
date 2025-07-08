import torch
import torch.nn as nn

from .lens_model import LensModel
from .source_model import SourceModel
from shared_utils import recursive_to_tensor


class LensingSystem(nn.Module):
    """
    Lensing system that handles multiple lens configurations in a batch
    """
    
    def __init__(self, num_samples, masses_data, precomputed_data, source_data, device, dtype=torch.float32):
        """
        Initialize the lensing system with multiple configurations
        
        Parameters:
            config_list: List of configuration dictionaries, one per lens system
            device: Device to run computations on
        """
        super().__init__()
        self.masses_data       = masses_data
        self.precomputed_data  = precomputed_data
        self.source_data       = source_data
        
        self. dtype = dtype
        self.device = device        
        
                        
        # Create a single lens model for all systems
        self.lens_model = LensModel(num_samples, masses_data, precomputed_data)
        
        # Create source models for each system
        self.source_model = SourceModel(source_data, precomputed_data)
        
            
                    
    def update_state(self, masses_data, precomputed_data, source_data):
        #saving only the initialization of super
        
        self.masses_data       = masses_data
        self.precomputed_data  = precomputed_data
        self.source_data       = source_data        
        # Create a single lens model for all systems
        self.lens_model = LensModel(masses_data, precomputed_data)
        # Create source models for each system
        self.source_models = SourceModel(source_data, precomputed_data)

        
    def forward(self, lens_grid):
        """
        Forward pass for all lens systems
        
        Parameters:
            lens_grid: Tensor of shape [B, H, W, 2] where B is the batch size (number of systems)
            
        Returns:
            Tensor of shape [B, H, W] with image plane intensities for all systems
        """
        if lens_grid.dtype != self.dtype:
            lens_grid = lens_grid.to(dtype=self.dtype)
            print("converting lens grid to dtype", self.dtype)
            
        # Get source plane positions for all systems
        source_grid = self.lens_model(lens_grid)
                
        
        
        source_evaluated=self.source_model(source_grid)
        
        return source_evaluated, source_grid
    