# dataset.py

import os
import torch
from torch.utils.data import Dataset
#from deep_learning.registry import DATASET_REGISTRY
from config import CATALOGS_DIR
from shared_utils import recursive_to_tensor, _grid_lens
from lensing_system import CatalogLensingSystem
from util import train_tf

#@DATASET_REGISTRY.register()
class NoNoiseDatasetJit(Dataset):
    def __init__(
        self,
        catalog_name=None,
        catalog_dict=None,
        samples_used="all",
        image_data_type=torch.float32,
        grid_width_arcsec=None,
        grid_pixel_side=None,
        final_transform=False,
        device=None
    ):
        super().__init__(catalog_name, catalog_dict, samples_used, image_data_type)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if grid_width_arcsec is None or grid_pixel_side is None:
            raise ValueError("grid_width_arcsec and grid_pixel_side must be provided")
        # 1) load full broadcast data
        file_path = os.path.join(CATALOGS_DIR, catalog_name + ".pth")
        raw = torch.load(file_path)
        masses_data     = raw["mass_components"]
        precomputed_data= raw["precomputed"]
        source_data     = raw["source_models"]
        # 2) build one big, static model
        self.model = torch.compile(
            CatalogLensingSystem(
                masses_data=masses_data,
                precomputed_data=precomputed_data,
                source_data=source_data,
                device=self.device,
                dtype=image_data_type
            )
        )
        # 3) grid stays fixed
        self.uncropped_grid = _grid_lens(grid_width_arcsec, grid_pixel_side, device=self.device)
        self.final_transform = final_transform

    def __len__(self):
        # fallback to base implementation
        return super().__len__()

    def get_batch(self, idxs):
        """
        idxs: List[int] or 1D LongTensor
        returns: (images [B,1,H,W], labels [B])
        """
        batch_idx = torch.as_tensor(idxs, device=self.device, dtype=torch.long)
        # forward‐only
        with torch.no_grad():
            # returns [B,H,W]
            imgs = self.model(self.uncropped_grid, batch_idx)
        # add channel
        imgs = imgs.unsqueeze(1)
        if self.final_transform:
            imgs = train_tf(imgs)
        # dynamic‐range adjust
        imgs = self.unit_max(imgs, use_log=False)
        # labels from catalog
        cat = recursive_to_tensor(self.catalog, self.device)
        num_sub = cat["SL_systems"]["lens_model"]["num_substructures"][batch_idx]
        labels = (num_sub > 0).long()
        return imgs, labels

    def __getitem__(self, idx):
        im, lb = self.get_batch([idx])
        return im[0], lb[0]
