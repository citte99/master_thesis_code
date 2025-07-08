from lensing_system import LensingSystem

from lensing_system import LensingSystem
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

class SensitivityMapMaker:
    def __init__(self, image_dict, substructure_to_test_dict, lens_grid, ResNet_obj, ResNet_params):
        """
        Initialize the SensitivityMapMaker.
        
        Parameters:
            image_dict: Dictionary containing the lensing system configuration
            substructure_to_test_dict: Dictionary containing the substructure to test
            lens_grid: Grid used for image generation (must match training grid)
            ResNet_obj: The ResNet model object
            ResNet_params: Parameters for the ResNet model
        """
        print("ATTENTION: the lens_grid must be the same provided during training. In the future, this will be forced somehow.")
        
        self.image_dict = image_dict
        self.substructure_to_test_dict = substructure_to_test_dict
        self.lens_grid = lens_grid
        self.ResNet_obj = ResNet_obj
        self.ResNet_params = ResNet_params
        
        # Determine device to use
        self.device = next(ResNet_obj.parameters()).device
        
        # Ensure lens_grid is on the correct device
        if self.lens_grid.device != self.device:
            self.lens_grid = self.lens_grid.to(self.device)
            
    def _create_clean_system(self):
        """Create a clean version of the lensing system without any substructures."""
        clean_system = copy.deepcopy(self.image_dict)
        
        # Filter out any substructures
        if "lens_model" in clean_system and "mass_components" in clean_system["lens_model"]:
            clean_system["lens_model"]["mass_components"] = [
                comp for comp in clean_system["lens_model"]["mass_components"] 
                if not comp.get("is_substructure", False)
            ]
            
            # Verify each component has required fields and handle both 'type' and 'mass_type'
            for comp in clean_system["lens_model"]["mass_components"]:
                # Map 'type' to 'mass_type' if needed
                if "type" in comp and "mass_type" not in comp:
                    comp["mass_type"] = comp["type"]
                
                if "mass_type" not in comp:
                    raise ValueError(f"Mass component missing both 'type' and 'mass_type': {comp}")
                if "params" not in comp:
                    raise ValueError(f"Mass component missing 'params': {comp}")
                    
            clean_system["lens_model"]["num_substructures"] = 0
        
        # Ensure precomputed field exists (required by LensingSystem)
        if "precomputed" not in clean_system:
            clean_system["precomputed"] = {}
            
        return clean_system

    def _add_substructure_at_position(self, clean_system, position):
        """Add the test substructure at a specified position."""
        system_with_sub = copy.deepcopy(clean_system)
        sub_copy = copy.deepcopy(self.substructure_to_test_dict)
        
        # Map 'type' to 'mass_type' in the substructure if needed
        if "type" in sub_copy and "mass_type" not in sub_copy:
            sub_copy["mass_type"] = sub_copy["type"]
        
        # Validate substructure has required fields
        if "mass_type" not in sub_copy:
            raise ValueError("Substructure dictionary must contain either 'type' or 'mass_type' field")
        if "params" not in sub_copy:
            raise ValueError("Substructure dictionary must contain 'params' field")
        if "is_substructure" not in sub_copy:
            sub_copy["is_substructure"] = True
        
        # Set the position in the substructure parameters
        sub_copy["params"]["pos"] = torch.tensor(position, device=self.device)
        
        # Add the substructure to the system
        system_with_sub["lens_model"]["mass_components"].append(sub_copy)
        system_with_sub["lens_model"]["num_substructures"] = 1
        
        return system_with_sub
    
    def create_sensitivity_map(self, resolution=20):
        """
        Create a sensitivity map by testing the substructure at different positions.
        
        Parameters:
            resolution: Number of positions to test in each dimension
            
        Returns:
            A 2D numpy array with probabilities of substructure detection
        """
        # Get clean system (no substructures)
        clean_system = self._create_clean_system()
        
        # Get grid extents
        x_min, x_max = self.lens_grid[:, 0].min().item(), self.lens_grid[:, 0].max().item()
        y_min, y_max = self.lens_grid[:, 1].min().item(), self.lens_grid[:, 1].max().item()
        
        # Create position grid
        x_positions = torch.linspace(x_min, x_max, resolution, device=self.device)
        y_positions = torch.linspace(y_min, y_max, resolution, device=self.device)
        
        # Initialize map
        sensitivity_map = torch.zeros((resolution, resolution), device=self.device)
        
        # Prepare model for evaluation
        self.ResNet_obj.eval()
        
        # Test each position
        for i in tqdm(range(resolution), desc="Creating sensitivity map"):
            for j in range(resolution):
                x_pos = x_positions[i]
                y_pos = y_positions[j]
                
                # Add substructure at current position
                test_system = self._add_substructure_at_position(clean_system, [x_pos, y_pos])
                
                # Create LensingSystem and generate image
                lensing_system = LensingSystem(test_system, self.device)
                
                with torch.no_grad():
                    image_tensor = lensing_system(self.lens_grid)
                
                # Ensure image has correct dimensions
                if image_tensor.ndim == 2:
                    image_tensor = image_tensor.unsqueeze(0)
                if image_tensor.ndim == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                
                # Get model's prediction
                with torch.no_grad():
                    output = self.ResNet_obj(image_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    # Get probability of substructure presence (class 1)
                    substructure_prob = probabilities[0, 1].item()
                
                # Store probability in sensitivity map
                sensitivity_map[j, i] = substructure_prob
        
        return sensitivity_map.cpu().numpy()
    
    def plot_sensitivity_map(self, sensitivity_map=None, resolution=20, save_path=None):
        """Plot the sensitivity map."""
        if sensitivity_map is None:
            sensitivity_map = self.create_sensitivity_map(resolution=resolution)
        
        # Get grid extents
        x_min, x_max = self.lens_grid[:, 0].min().item(), self.lens_grid[:, 0].max().item()
        y_min, y_max = self.lens_grid[:, 1].min().item(), self.lens_grid[:, 1].max().item()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot map
        im = ax.imshow(sensitivity_map, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                       cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar and labels
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability of Substructure Detection')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Sensitivity Map for Substructure Detection')
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_sensitivity_map(self, path, sensitivity_map=None, resolution=20):
        """Save the sensitivity map to a file."""
        if sensitivity_map is None:
            sensitivity_map = self.create_sensitivity_map(resolution=resolution)
        
        np.save(path, sensitivity_map)
        return path