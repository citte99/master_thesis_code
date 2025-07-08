# Datasets/custom_datasets/dataset_a.py
import torch
from typing import List, Tuple
from ..base import CustomDatasetBase
from catalog_manager import CatalogManager
from deep_learning.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class DatasetA(CustomDatasetBase):
    """
    Example implementation of a custom dataset.

    THIS IS OUTDATED, WHAT NO NOISE DATASET INSTEAD.
    """
    def __init__(self, catalog_name=None, catalog_dict=None, samples_used="all", image_data_type=torch.float32, transform=None):
        raise NotImplementedError("This dataset is provided as an example only.")


        super().__init__(catalog_name, catalog_dict, samples_used, image_data_type)
        self.transform = transform
        self.catalog_manager = CatalogManager(self.catalog_name)
        
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
        
        for idx in idxs:
            # Get data from catalog manager
            img, label = self.catalog_manager.get_item(idx)
            
            # Apply transformations if needed
            if self.transform:
                img = self.transform(img)
                
            # Convert to appropriate tensor type
            img_tensor = torch.tensor(img, dtype=self.image_data_type)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            images.append(img_tensor)
            labels.append(label_tensor)
        
        # Stack tensors into batches
        images_batch = torch.stack(images)
        labels_batch = torch.stack(labels)
        
        return images_batch, labels_batch