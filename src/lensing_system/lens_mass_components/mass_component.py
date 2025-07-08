import torch
import torch.nn as nn

class MassComponent(nn.Module):
    def __init__(self):
        """
            We unpack the tensor configuration in the common features
            between different mass components.

            In this code, all the components are at same redshift, so the
            redshift is not a parameter of the mass component.

            We have only the position.
        """
        super().__init__()


    def deflection_angle(self, lens_grid, z_source, precompt_dict):
        """
           This forward computes the deflection field of the mass component
        """
        pass