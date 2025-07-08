# Standardization of the datasets

For ease of use in the other functionalities of the code, I here standardize the architecture of the datasets.

The key caracteristics are:
- the noise is applied at the inheriter level (inheriting form CustomDataset).
- the datasets implement the function get_batch, either in case they compute batched operations or not.
- The get batch provides images and labels already stacked.

The dataloader class is made to work with the __getitem__ of the dataset, so we need to modify the standard collate function.

Normally, dataloader uses a index sampler, gets the random indexes if shuffle is true, calls the get item on the selected indexes,
and then gives the images and labels to the collate that creates the (lists? tuples? whatever).

When provided with a batch sampler, it no longer calls the getitem, but it passes the indexes to the collate function. 
(depending on which sampler). THE DISTINCION IN WHAT IS PASSED TO THE COLLATE IS WHETHER YOU GIVE TO THE DATALOADER THE BATCH SIZE.

``` python

import torch
from torch.utils.data import Dataset, Dataloader, Sampler
from typing import List, Tuple, Optional
from catalog_manager import CatalogManager

class CustomDatasetBase(Dataset):
    """
        The noise, the transform, and all particular stuff is performed by the inheriters
    """
    def __init__(self, catalog_name, samples_used= "all", image_data_type=torch.float32):


        self.catalog = catalog
        self.samples_used = samples_used
        self.image_data_type = image_data_type


        #the check that the catalog exists is performed by the catalog manager.
        catalog_len=CatalogManager(self.catalog_name).len()
        if sample_used == "all":
            self.len=catalog_len
        elif sample_used > catalog_len:
            raise ValueError("The sample_used parameter is bigger than this catalog's number of systems.")
        else:
            self.len=sample_used
            
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        raise NotImplementedError("The get item must not be implemented, we use the get_batch")

    def get_batch(self, idxs: List[int])-> Tuple[List[torch.tensor], List[int]]: #maybe bool?
        pass



class RandomBatchIndexSampler(Sampler):
    def __init__(self, my_dataset, batch_size, shuffle=True):
        self.len_dataset=my_dataset.__len__()
        self.batch_size=batch_size
        self.shuffle=shuffle

    def __iter__(self):
        indices=list(range(self.len_dataset))

        if self.shuffle:
            generator=torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            torch.randperm(self.len_dataset, generator=generator, out=torch.LongTensor(indices))

        for i in range(0, self.len_dataset, self.batch_size):
            yield indices[i, i+self.batch_size]

    def __len__(self):
        return (len(self.len_dataset) + self.batch_size - 1) // self.batch_size



def custom_collate_fn(batch_indices, dataset):
    #the dataloader strangely wraps the list from the batch index sampler in another list.
    images, labels = dataset.get_batch(batch_indices[0])
    #the images and labels are provided already stacked by the get_batch method.
    return images, labels



def custom_dataloader(dataset, batch_size=32, shuffle=True, **kwargs):
    sampler = RandomBatchSampler(dataset, batch_size, shuffle)

    return Dataloader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda x: get_batch_collate(x, dataset),
        **{k: v for k, v in kwargs.items() if k not in ['batch_size', 'shuffle']}

    )


# Example usage
from Datasets import custom_dataloader
from Datasets.custom_datasets import DatasetA

# Create dataset
dataset = DatasetA(catalog_name="my_catalog", samples_used="all")

# Create dataloader
my_dataloader = custom_dataloader(dataset, batch_size=4)

# Iterate through batches
for batch_idx, (images, labels) in enumerate(my_dataloader):
    print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")
    # Training code would go here
    
    # Just show first 3 batches
    if batch_idx >= 2:
        break


```