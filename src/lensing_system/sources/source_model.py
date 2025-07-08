import torch.nn as nn

class SourceModel(nn.Module):
    """
    The only initial constraint is the redshift of the source.
    """

    def __init__(self):
        super().__init__()

    def forward(self, source_grid):
        """
        Compute the brightness of the source on the source grid.
NOTE
        """
        raise NotImplementedError
