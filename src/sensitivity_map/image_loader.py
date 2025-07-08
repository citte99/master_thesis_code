import torch
import torch.nn as nn

from .lens_model import LensModel
from .source_model import SourceModel
from shared_utils import recursive_to_tensor
from .lens_mass_components import SIS, NFW, ExternalPotential, PEMD


class SensMapImageLoader(nn.Module):
    """
    Lensing system that handles multiple lens configurations in a batch
    """
    
    def __init__(self, main_masses_data, substructure_data, precomputed_data, source_data, device, dtype=torch.float32):
        """
        Initialize the lensing system with multiple configurations
        
        Parameters:
            config_list: List of configuration dictionaries, one per lens system
            device: Device to run computations on
        """
        super().__init__()
        self.masses_data       = masses_data       # this is the tensor for a single sistem
        self.substructure_data = substructure_data # this is the tensor representing all the positions/whatever of the sub
        self.precomputed_data  = precomputed_data
        self.source_data       = source_data
        
        self. dtype = dtype
        self.device = device
        
        """
        correct this
        """
        self.tot_images=substructure_data.len() 
        self.last_computed_image_index=0
                        
        # Create a single lens model for all systems
        self.lens_model_sub = LensModel(masses_data, precomputed_data)
        
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

        
    def get_batch(self, lens_grid, batch_size):
        """
        Here all the data of the substructure is loaded once, so instead of a single forward, I need a get_batch
        """
        if lens_grid.dtype != self.dtype:
            lens_grid = lens_grid.to(dtype=self.dtype)
            print("converting lens grid to dtype", self.dtype)
        
    
        #take the right slice of the substructure_data
        last_idx=self.last_computed_index
        
        if last_idx=self.tot_images:
            raise StopIteration
            
        elif:
            batch_size+last_idx<self.tot_images:
            sub_data=self.substructure_data[last_idx+1 : last_idx+batch_size+1]
            last_idx=last_idx+batch_size+1
        else:
            sub_data=self.substructure_data[last_idx+1 : ]
            last_idx=self.tot_images
            
        
        self.lens_model_sub.update_sub_data(sub_data)
    
        source_grid = self.lens_model(lens_grid)
                
        
        
        source_evaluated=self.source_model(source_grid)
        
        return source_evaluated, source_grid
    
    
    
    
    
    
    
    
class LensModel(nn.Module):
    """
    Batched lens model using prestructured flat-catalog data.
    """
    def __init__(self, masses_data: dict, precomputed: dict, device="cuda", dtype=torch.float32):
        """
        masses_data:                                    precomputed:
        #type=PEMD    ....    #type=SIS ....     
        |i| |P|                                         same.
        |n| |a|
        |d| |r|
        |e| |a|
        |x| |m|
        |e| |s|
        |s| | |


        The total number of systems can be extrapolated form the max index assigned to any mass component.
        Here we pass it for clarity, and as a check.
        We get the deflection grids for each mass component, and we need to sum them per-same-index.

        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.main_lens_deflection=None
        # Precomputed distances etc.
        self.precomputed = precomputed
        precomp_map = precomputed["param_map"]

        # Map from component type to class
        self.component_classes = {
            "SIS": SIS,
            "NFW": NFW,
            "ExternalPotential": ExternalPotential,
            "PEMD": PEMD,
        }


        self.tot_number_of_systems = number_of_systems
        # Could check if the indexes of the masses make sense




        # Build dict of instantiated components and their system indices
        self.components = {}

        self.system_indices = {}

        for comp_type, comp_dat in masses_data.items():
            # Tensor of shape [M, K]
            params = comp_dat['params'].to(device, dtype)
            idxs   = comp_dat['sys_idx'].to(device, torch.long)

            if comp_type not in self.component_classes:
                raise ValueError(f"Unknown component type: {comp_type}")

            # Instantiate the component with its parameter tensor
            cls = self.component_classes[comp_type]

            # precomputed data must be adjusted to the indexes of the mass component
            this_comp_indexes = comp_dat['sys_idx']
            this_comp_precomp_params = precomputed["params"][this_comp_indexes]
            this_comp_precomp_index = precomputed["sys_idx"][this_comp_indexes]
            precomp_init={
                "params": this_comp_precomp_params,
                "param_map": precomp_map,
                "sys_idx": this_comp_precomp_index
            }

            self.components[comp_type] = cls(params, precomp_init, device=device, dtype=dtype)
            self.system_indices[comp_type] = idxs

            
    def update_sub_data(self, sub_data):
        
            
            
    def deflection_field_main_lens(self, lens_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute total deflection for each batch sample.
        """
        
        # If already calculated, return it
        if self.main_lens_deflection is not None:
            return self.main_lens_deflection

        total_defl = torch.zeros(
            (self.tot_number_of_systems, *lens_grid.shape), 
            dtype=self.dtype, 
            device=lens_grid.device
        )   
        


        for comp_type, component in self.components.items():
            idxs = self.system_indices[comp_type]
            # select the grid points for this component's systems
            
            # I think I do not need to expand the grid
            defl = component.deflection_angle(lens_grid)


            # I get from this torch.Size([100, 100, 100, 2]),
            # which is good. Each deflection belongs to the system of the indexes
            #total_defl[idxs] = defl[idxs] is not fine because there are multiple systems belonging to the same index


            # scatter-add back into total
            total_defl.index_add_(0, idxs, defl)
        self.main_lens_deflection=total_defl
        return total_defl
    
    
     def deflection_field_sub(self, lens_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute total deflection for each batch sample.
        """

        total_defl = torch.zeros(
            (self.tot_number_of_systems, *lens_grid.shape), 
            dtype=self.dtype, 
            device=lens_grid.device
        )   
        


        for comp_type, component in self.components.items():
            idxs = self.system_indices[comp_type]
            # select the grid points for this component's systems
            
            # I think I do not need to expand the grid
            defl = component.deflection_angle(lens_grid)


            # I get from this torch.Size([100, 100, 100, 2]),
            # which is good. Each deflection belongs to the system of the indexes
            #total_defl[idxs] = defl[idxs] is not fine because there are multiple systems belonging to the same index


            # scatter-add back into total
            total_defl.index_add_(0, idxs, defl)
        return total_defl

    def forward(self, lens_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute source plane positions: lens_grid - deflection_field
        """
        #the structure not to reapeat the calculation is in the method
        main_lens_deflection=self.deflectin_field_main_lens(lens_grid)
        
        deflection_sub=self.deflection_field_sub(lens_grid)
        
        source_grid= lens_grid - (main_lens_deflection+deflection_sub)

    
        return source_grid

