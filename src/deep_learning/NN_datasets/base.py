# Datasets/base.py
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from catalog_manager import CatalogManager
from shared_utils import recursive_to_tensor, _grid_lens
import os
from config import CATALOGS_DIR


class CustomDatasetBase(Dataset):
    """
    The base dataset class. Noise, transformations, and other specific functionality
    should be implemented by inheriting classes.
    """
    def __init__(self, catalog_name=None, catalog_dict=None, samples_used="all",broadcasting=False, image_data_type=torch.float32, config=None):
        self.catalog_name = catalog_name
        self.samples_used = samples_used
        self.image_data_type = image_data_type

        # Check that the catalog exists via the catalog manager
        if catalog_name is not None and catalog_dict is not None:
            raise ValueError("You cannot provide both a catalog name and a catalog dictionary. Please provide one or the other.")
        elif catalog_dict is not None:
            # Use the provided catalog dictionary
            self.catalog = catalog_dict
            import json
            self.len = len(catalog_dict["SL_systems"])
        else:
        # Set the attribute len and perform slicing 
            
            if broadcasting==False:
                catalog_manager=CatalogManager(self.catalog_name)
                self.catalog = catalog_manager.catalog
                catalog_len = catalog_manager.len()

                if samples_used == "all":
                    self.len = catalog_len
                elif samples_used > catalog_len:
                    raise ValueError("The samples_used parameter is bigger than this catalog's number of systems.")
                else:
                    self.len = samples_used

                self.catalog["SL_systems"]= self.catalog["SL_systems"][:self.len]
                
            elif broadcasting==True:
                # we need the file pth. 
                #if there is a final json remove it
                
                file_path = os.path.join(CATALOGS_DIR, (catalog_name+".pth"))
                self.data=torch.load(file_path)
                print("Using broadcasting mode")
                
                
                precomputed_data = self.data['precomputed']
                
                
                index_max=torch.max(precomputed_data['sys_idx'])
                
                
                if samples_used=="all":
                    self.len=index_max+1
                elif samples_used> index_max+1:
                    raise ValueError("The samples_used parameter is bigger than this catalog's number of systems.")
                else:
                    self.len=samples_used
                    
                m = 0
                n=self.len
                
                    
                mask= (precomputed_data['sys_idx']>=m)& (precomputed_data['sys_idx']< m+n)
                precomputed_data['sys_idx'] = precomputed_data['sys_idx'][mask].to(device='cuda:0')
                precomputed_data['params'] = precomputed_data['params'][mask].to(device='cuda:0')
                
                                        
                labels_data=self.data['labels']
                mask= (labels_data['sys_idx']>=m)& (labels_data['sys_idx']< m+n)
                labels_data['sys_idx'] = labels_data['sys_idx'][mask].to(device='cuda:0')
                labels_data['label_values'] = labels_data['label_values'][mask].to(device='cuda:0')

                

                
                masses_data = self.data['mass_components']
                
                #total number of systems is taken as the maxiumum index of the mass components
                #now create a mask for sys idx selecting only the desired systems, and mask both
                #the indexes and the params
                
                #these are set up like this for changing to selecting a slice
               
                


                for mtype, data in masses_data.items():
                    # Convert to CUDA tensors
                    mask= (masses_data[mtype]['sys_idx']>=m) & (masses_data[mtype]['sys_idx']< m+n)
                    masses_data[mtype]['params'] = data['params'].to(device='cuda:0')
                    masses_data[mtype]['sys_idx'] = data['sys_idx'].to(device='cuda:0')
                    
                
                                        
                #now we need to filter for the first samples_used indexes
                
               
                                        
                                        
                                        
                source_data = self.data['source_models']    

                for stype, data in source_data.items():
                    # Convert to CUDA tensors
                    mask= (source_data[stype]['sys_idx']>=m) & (source_data[stype]['sys_idx']< m+n)
                    source_data[stype]['params'] = data['params'][mask].to(device='cuda:0')
                    source_data[stype]['sys_idx'] = data['sys_idx'][mask].to(device='cuda:0')

                self.data={
                    'mass_components': masses_data,
                    'precomputed': precomputed_data,
                    'source_models': source_data,
                    'labels': labels_data
                }

                

            if config is not None:
                raise AttributeError("If you got here, it means you provided a configuration for a dataloader that does not need it. Please remove it.")

    def update_catalog_dict(self, catalog_dict):
        """
        Update the catalog dictionary with a new one.
        This method is intended to not re-initialize the dataset, but to update the catalog
        
        Args:
            catalog_dict: New catalog dictionary to use
        """
        self.catalog = catalog_dict
        self.len = len(catalog_dict["SL_systems"])
        device = "cuda"
        self.catalog = recursive_to_tensor(self.catalog, device)



    def __len__(self):
        print(self.len)
        return self.len

    def __getitem__(self, idx):
        raise NotImplementedError("The get item must not be implemented, we use the get_batch")

    def get_batch(self, idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of samples indexed by idxs.
        Must be implemented by child classes.
        
        Args:
            idxs: List of indices to retrieve
            
        Returns:
            Tuple of (images, labels) as tensors
        """
        raise NotImplementedError("Child classes must implement get_batch")