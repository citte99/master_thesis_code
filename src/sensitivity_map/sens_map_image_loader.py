import torch
import torch.nn as nn

#from lensing_system_broadcasting import LensModel
#from lensing_system_broadcasting import SourceModel

from lensing_system_broadcasting.sources import GaussianBlob #Sersic  # add other source classes as needed

from shared_utils import recursive_to_tensor
from lensing_system_broadcasting.lens_mass_components import SIS, NFW, ExternalPotential, PEMD


class SensMapImageLoader(nn.Module):

    def __init__(self, main_masses_data, substructure_data, precomputed_data, source_data, lens_grid, device, dtype=torch.float32):
        """
        main_masses_data has the same structure as in lensing system broadcasting, but all the indexes are 0 as it is only one system.
        
        Same is for precomputed_data and source data.
        
        substructure data is a dictionary having keys ["pix_tuple", "index_abs", "pix_idx", "mass_type", "params", "params_map"]
        """
        super().__init__()
        self.masses_data       = main_masses_data       # this is the tensor for a single sistem
        self.substructure_data = substructure_data # this is the tensor representing all the positions/whatever of the sub
        self.precomputed_data  = precomputed_data
        self.source_data       = source_data
        
        self. dtype = dtype
        self.device = device
        
        
        if lens_grid.dtype != self.dtype:
            self.lens_grid = lens_grid.to(dtype=self.dtype)
            print("converting lens grid to dtype", self.dtype)
        else:
            self.lens_grid=lens_grid
        """
        correct this
        """
        self.tot_images=len(substructure_data["index_abs"])
                        
        # Create a single lens model for all systems
        self.lens_model       = LensModel(self.masses_data, self.precomputed_data, self.substructure_data)
        
        # Create source models for each system
        self.source_model = SourceModel(source_data, precomputed_data)
        
            
                    
        
    def __getitem__(self, my_slice):
        
        # detect slicing
        if isinstance(my_slice, slice):
            
            
            source_grid = self.lens_model(self.lens_grid, my_slice)
            
            #print(source_grid.shape)



            source_evaluated=self.source_model(source_grid)
            #print(source_evaluated.shape)

            return source_evaluated, source_grid


        else:
            raise ValueError("Only accepting slices right now")
            
            
            
        """
        Here all the data of the substructure is loaded once, so instead of a single forward, I need a get_batch
        """
        
    
    
    
    
    
    
    
class LensModel(nn.Module):
    """
    Batched lens model using prestructured flat-catalog data.
    """
    def __init__(self, masses_data: dict, precomputed: dict,sub_data: dict,  device="cuda", dtype=torch.float32):
        
        
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
        self.tot_number_of_systems = 1  # or derive from your data
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

        self.sub_data=sub_data
        
        
        # Build dict of instantiated components and their system indices
        self.components = {}

        self.system_indices = {}

        for comp_type, comp_dat in masses_data.items():
            # Tensor of shape [M, K]
            params = comp_dat['params'].to(device, self.dtype)
            # check that the indexes are all 0 in this case
            if len(set(comp_dat['sys_idx']))>1:
                raise ValueError

            if comp_type not in self.component_classes:
                raise ValueError(f"Unknown component type: {comp_type}")

            # Instantiate the component with its parameter tensor
            cls = self.component_classes[comp_type]

            # precomputed data must be adjusted to the indexes of the mass component
            
            this_comp_precomp_params = precomputed["params"]
            
            precomp_init={
                "params": this_comp_precomp_params,
                "param_map": precomp_map,
                "sys_idx": torch.tensor([0], device="cuda")
            }

            self.components[comp_type] = cls(params, precomp_init, device=device, dtype=self.dtype)
            self.system_indices[comp_type] = torch.tensor([0], device="cuda")
            

            
            
            
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
    
    
    def deflection_field_sub(self, lens_grid: torch.Tensor, my_slice) -> torch.Tensor:
        # assume all subs are same type
        idxs = torch.arange(my_slice.start, my_slice.stop, device=self.device, dtype=torch.long)
        n_sub = idxs.numel()

        # slice out only the per‐sub info
        sliced_params = self.sub_data["params"][idxs]  # (n_sub, P)

        # tile the precomputed params n_sub times along a new batch dim
        precomp_parms = self.precomputed["params"]

        # build the init dict correctly
        precomp_init = {
            "params": precomp_parms,
            "param_map": self.precomputed["param_map"],
            "sys_idx": torch.arange(n_sub, device=self.device, dtype=torch.long)
        }

        # instantiate locally (don’t stash on self)
        SubClass = self.component_classes[self.sub_data["mass_type"]]
        sub_instance = SubClass(sliced_params, precomp_init,
                                device=self.device, dtype=self.dtype)

        # make sure lens_grid has a batch dim
        #grid = lens_grid.unsqueeze(0).expand(n_sub, *lens_grid.shape)
        sub_defl = sub_instance.deflection_angle(lens_grid)  # should be (n_sub, Ny, Nx, 2)

        
        print(f"sub defl shape: {sub_defl.shape}")
        
        print(f"sub defl avg per item: {sub_defl.mean((1,2,3))}")

        
        return sub_defl




    def forward(self, lens_grid: torch.Tensor, my_slice) -> torch.Tensor:
        """
        Compute source plane positions: lens_grid - deflection_field
        """
        #the structure not to reapeat the calculation is in the method
        main_lens_deflection=self.deflection_field_main_lens(lens_grid)
        
        deflection_sub=self.deflection_field_sub(lens_grid, my_slice)
        
        source_grid= lens_grid - (main_lens_deflection+deflection_sub)
        
        
#         print(f"souce grid shape: {source_grid.shape}")
#         print(f"souce grid avg: {source_grid.mean((1,2,3))}")
        
#         import matplotlib.pyplot as plt
#         for grid in source_grid:
#             plt.scatter(grid.detach().cpu()[:, :, 0], grid.detach().cpu()[:, :, 1])
#             plt.show()
            
        
        return source_grid

    
    

class SourceModel(nn.Module):
    """
    Batched source model using prestructured flat-catalog data.
    """
    def __init__(self, source_data: dict, precomputed: dict, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.source_data=source_data
        self.precomputed=precomputed

        # Map from source type key to the source class implementation
        self.source_classes = {
            "Gaussian_blob": GaussianBlob,
        #    "Sersic": Sersic,
            # add other source types here
        }

        # Instantiate one component per source type
        self.components = {}
        self.system_indices = {}
        

    def forward(self, source_grid: torch.Tensor) -> torch.Tensor:
        """
        Evaluate and sum all source contributions on the source_grid.

        Args:
            source_grid: [B, H, W, 2]

        Returns:
            brightness: Tensor of shape [B, H, W]
        """
        
        
        
        
        if source_grid.dtype != self.dtype:
            source_grid = source_grid.to(dtype=self.dtype)

        B, H, W, _ = source_grid.shape
        output = torch.zeros((B, H, W), device=self.device, dtype=self.dtype)
        
        
        
        for stype, dat in self.source_data.items():
            if stype not in self.source_classes:
                raise ValueError(f"Unknown source model type: {stype}")
            # params tensor has shape [M, K]
            params = dat['params'].repeat(B, 1)
            # which system each row belongs to
            #idxs   = dat['sys_idx']

            # instantiate the source class; each should accept (param_tensor, precomputed, device, dtype)
            cls = self.source_classes[stype]
            self.components[stype] = cls(params, self.precomputed)
            #self.system_indices[stype] = idxs
        

        # loop over each source type
        for stype, component in self.components.items():
            idxs = torch.arange(0, B, device="cuda")
            subgrid = source_grid[idxs]        # [M, H, W, 2]
            vals    = component(subgrid)       # [M, H, W]
            output.index_add_(0, idxs, vals)

        return output