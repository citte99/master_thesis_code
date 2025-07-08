import torch
import torch.nn as nn

from .lens_mass_components import SIS, NFW, ExternalPotential, PEMD

class LensModel(nn.Module):
    """
    Batched lens model using prestructured flat-catalog data.
    """
    def __init__(self, number_of_systems:int, masses_data: dict, precomputed: dict, device="cuda", dtype=torch.float32):
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

    def deflection_field(self, lens_grid: torch.Tensor) -> torch.Tensor:
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
        source_grid= lens_grid - self.deflection_field(lens_grid)

    
        return source_grid

