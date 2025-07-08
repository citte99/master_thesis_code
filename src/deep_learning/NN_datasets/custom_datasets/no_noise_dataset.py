import torch
import torchvision.transforms as T
from typing import List, Tuple
from ..base import CustomDatasetBase
from catalog_manager import CatalogManager
from shared_utils import recursive_to_tensor, _grid_lens
from lensing_system import LensingSystem
from lensing_system_broadcasting import LensingSystem as LensingSystemBroadcasting
from deep_learning.registry import DATASET_REGISTRY
from config import CATALOGS_DIR
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


from .util import train_tf


@DATASET_REGISTRY.register()
class NoNoiseDataset(CustomDatasetBase):
    """
    Example implementation of a custom dataset.
    """
    def __init__(self, catalog_name=None, catalog_dict=None, samples_used="all", image_data_type=torch.float32, grid_width_arcsec=None, grid_pixel_side=None, upscaling=None, final_transform=False, broadcasting=False, device=None):

        super().__init__(catalog_name, 
                         catalog_dict=catalog_dict, 
                         samples_used=samples_used, 
                         image_data_type=image_data_type, 
                         broadcasting=broadcasting)    
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.upscaling=upscaling
        
        
        
        
        if grid_width_arcsec is None or grid_pixel_side is None:    
            raise ValueError("grid_width_arcsec and grid_pixel_side must be provided")
            
        if self.upscaling:
            self.uncropped_grid = _grid_lens(grid_width_arcsec, grid_pixel_side*upscaling, device=self.device)
            
        else:
            self.uncropped_grid = _grid_lens(grid_width_arcsec, grid_pixel_side, device=self.device)

        self.final_transform = final_transform
        self.broadcasting = broadcasting
        print(f"Using device: {self.device}")
            

        if self.broadcasting==False:
            self.catalog = recursive_to_tensor(self.catalog, self.device)
        
    
        print("Currently this dataloader is calculating the images in float32")

    # def __len__(self): the len is defined in the base class
    
    def unit_max(self, imgs, eps=1e-6, use_log=False):
        """
        imgs: Tensor [B, 1, H, W]  (positive floats)

        1. per-image divide by brightest pixel  → in [0, 1]
        2. optional log1p compression          → shrinks dynamic range
        """
        # 1️⃣  divide by brightest (amax over H×W – keep dims for broadcasting)
        max_val = imgs.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
        imgs = imgs / max_val

        # 2️⃣  optional: compress very bright cores
        
        if use_log:
            imgs = torch.log1p(imgs * 9) / math.log(10)   # still ≈[0,1]

        return imgs

    def get_batch(self, idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of samples indexed by idxs.
        
        Args:
            idxs: List of indices to retrieve
            
        Returns:
            Tuple of (images, labels) as tensors
        """
        images = []
        labels = []
        if not self.broadcasting:
            for idx in idxs:
                # Get data from catalog manager
                system_dict_tensor = self.catalog["SL_systems"][idx]
                num_sub = system_dict_tensor["lens_model"]["num_substructures"]
                label_tensor = torch.tensor((num_sub > 0)).to(device=self.device).long()
                
                lensing_system = LensingSystem(system_dict_tensor, self.device)
                

                with torch.no_grad():
                    image_tensor = lensing_system(self.uncropped_grid)
                    # imageplot1= image_tensor.squeeze(0).cpu()
                    # plt.imshow(imageplot1)
                    # plt.show()

            
                # Convert to appropriate tensor type
                img_tensor = image_tensor.to(dtype=self.image_data_type)
                label_tensor = label_tensor.to(dtype=torch.long)
                
#                 imageplot2= img_tensor.squeeze(0).cpu()
#                 plt.imshow(imageplot2)
#                 plt.show()
        
                
                images.append(img_tensor)
                labels.append(label_tensor)
            
            # Stack tensors into batches
            images_batch = torch.stack(images)
            if self.upscaling is not None:
                images_batch=F.avg_pool2d(images_batch, kernel_size=self.upscaling, stride=self.upscaling)
            
            labels_batch = torch.stack(labels)
        else:
            # idxs: a 1D list or tensor of the desired system‐indices in this batch
            # e.g. idxs = [2, 5, 6] or torch.tensor([2,5,6])
            # ensure we have a LongTensor on the right device
            if isinstance(idxs, torch.Tensor):
                selected_idxs = idxs.to(self.device).long()
            else:
                selected_idxs = torch.as_tensor(idxs, device=self.device, dtype=torch.long)

            batch_size = selected_idxs.numel()

            masses_data = self.data['mass_components']
            precomputed_data = self.data['precomputed']
            source_data = self.data['source_models']    
            label_data=self.data['labels']

            # 1) subset each mass‐component
            sub_masses = {}
            for mtype, data in masses_data.items():
                params   = data['params']    # (N_comp, …)
                sys_idx  = data['sys_idx']   # (N_comp,)
                # mask all entries whose sys_idx ∈ selected_idxs
                mask     = (sys_idx.unsqueeze(1) == selected_idxs.unsqueeze(0)).any(dim=1)
                p_sub    = params[mask]
                idx_sub  = sys_idx[mask]
                # remap old idx -> 0..batch_size-1
                new_idx  = (idx_sub.unsqueeze(1) == selected_idxs.unsqueeze(0)).float().argmax(dim=1)
                sub_masses[mtype] = dict(params=p_sub, sys_idx=new_idx)

            # 2) subset the precomputed lookup table
            pc_params = precomputed_data['params']
            pc_idx    = precomputed_data['sys_idx']
            mask_pc   = (pc_idx.unsqueeze(1) == selected_idxs.unsqueeze(0)).any(dim=1)
            pc_p_sub  = pc_params[mask_pc]
            pc_i_sub  = pc_idx[mask_pc]
            pc_new_idx = (pc_i_sub.unsqueeze(1) == selected_idxs.unsqueeze(0)).float().argmax(dim=1)
            sub_precomputed = dict(params=pc_p_sub, sys_idx=pc_new_idx, param_map=precomputed_data['param_map'])

            # 3) subset each source‐model
            sub_sources = {}
            for stype, data in source_data.items():
                params   = data['params']
                sys_idx  = data['sys_idx']
                mask     = (sys_idx.unsqueeze(1) == selected_idxs.unsqueeze(0)).any(dim=1)
                p_sub    = params[mask]
                idx_sub  = sys_idx[mask]
                new_idx  = (idx_sub.unsqueeze(1) == selected_idxs.unsqueeze(0)).float().argmax(dim=1)
                sub_sources[stype] = dict(params=p_sub, sys_idx=new_idx)
            
            #subset the labels
            label_values =label_data['label_values']
            label_idx    =label_data['sys_idx']
            mask_lab  = (label_idx.unsqueeze(1) == selected_idxs.unsqueeze(0)).any(dim=1)
            lab_val_sub=label_values[mask_lab]
            lab_i_sub=label_idx[mask_lab]
            lab_new_idx=(lab_i_sub.unsqueeze(1) == selected_idxs.unsqueeze(0)).float().argmax(dim=1)
            #sub_lab=dict(label_values=lab_val_sub, sys_idx= lab_new_idx)

            
            
            
            # 4) build batch‐config and run
            configs = {
                'num_samples': batch_size,
                'masses_data': sub_masses,
                'precomputed_data': sub_precomputed,
                'source_data': sub_sources
            }
            lensing_system = LensingSystemBroadcasting(
                **configs,
                device=self.device,
                dtype=torch.float32
            )


            images_batch = lensing_system(self.uncropped_grid)[0]      # -> (batch_size, H, W) or similar

            # 5) labels: probably this should be prealloacated
            lab_val_sub = lab_val_sub.squeeze(-1)       # -> (batch_size,)

            # now create and fill your labels_batch
            labels_batch = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            labels_batch[lab_new_idx] = lab_val_sub.to(dtype=torch.long, device=self.device)



            
        # 6) add a channel dimension to the images    
        
        
        images_batch= images_batch.unsqueeze(1) 
        
        if self.upscaling is not None:
                images_batch=F.avg_pool2d(images_batch, kernel_size=self.upscaling, stride=self.upscaling)
        
        # Add random crop to the images
        if self.final_transform:
            images_batch = train_tf(images_batch)
        
#         else:
#             print("Currently not cropping and rotating, set final_transform in the intialization to True to use it")
        
        
#         # put pixels in [0,1] if they aren't already
#         images_batch = images_batch.float() / 255.0  

#         # zero-centre & scale:  (x - 0.5) / 0.5
#         #images_batch = T.functional.normalize(images_batch, mean=[0.5], std=[0.5])

        images_batch = self.unit_max(images_batch, use_log=False)   # or True


        return images_batch, labels_batch