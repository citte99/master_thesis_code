# Datasets/dataloaders.py
from torch.utils.data import DataLoader
from .samplers import RandomBatchSampler

# Datasets/dataloaders.py
from torch.utils.data import DataLoader
from .samplers import RandomBatchSampler

def custom_collate_fn(batch_indices, dataset):
    # batch_indices is a list of indices, but we want to handle the whole batch at once
    # We don't need the nested list that DataLoader would normally create
    flat_indices = [idx for sublist in batch_indices for idx in sublist]
    images, labels = dataset.get_batch(flat_indices)
    return images, labels

def custom_dataloader(dataset, batch_size=32, shuffle=True, **kwargs):
    """
    Creates a custom dataloader for CustomDatasetBase datasets that bypasses __getitem__
    and directly uses get_batch.
    
    Args:
        dataset: A dataset inheriting from CustomDatasetBase
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        DataLoader instance configured for the custom dataset
    """
    # Create a dummy dataset that just returns indices
    # This way, __getitem__ is never called on our actual dataset
    class IndexDataset:
        def __init__(self, length):
            self.length = length
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            return idx
    
    index_dataset = IndexDataset(len(dataset))
    sampler = RandomBatchSampler(index_dataset, batch_size, shuffle)
    
    return DataLoader(
        index_dataset,
        batch_sampler=sampler,
        collate_fn=lambda indices: dataset.get_batch(indices),
        num_workers=0,
        **{k: v for k, v in kwargs.items() if k not in ['batch_size', 'shuffle']}
    )

# import torch
# from torch.utils.data import DataLoader
# from .samplers import RandomBatchSampler

# def custom_dataloader(dataset, batch_size=32, shuffle=True, device=None, **kwargs):
#     device = device if device is not None else dataset.device

#     # build an index‐only dataset
#     class IndexDataset:
#         def __init__(self, length):  self.length = length
#         def __len__(self):          return self.length
#         def __getitem__(self, idx): return idx

#     index_ds = IndexDataset(len(dataset))
#     sampler   = RandomBatchSampler(index_ds, batch_size, shuffle)

#     # collate_fn now does exactly one as_tensor per batch
#     def collate_fn(batch_indices):
#         # batch_indices is a list of Python ints
#         idxs = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
#         return dataset.get_batch(idxs)

#     return DataLoader(
#         index_ds,
#         batch_sampler=sampler,
#         collate_fn=collate_fn,
#         num_workers=0,
#         **{k: v for k, v in kwargs.items() if k not in ("batch_size","shuffle")}
#     )

from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler

def distributed_dataloader(dataset, batch_size=32, shuffle=False, **kwargs):
    """
    Like custom_dataloader, but shards batches across ranks via DistributedSampler.
    """
    # 1) Build an index-only dataset
    class IndexDataset:
        def __init__(self, length):
            self.length = length
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            return idx

    index_dataset = IndexDataset(len(dataset))

    # 2) Create a DistributedSampler on that index‐only dataset
    #    By default it will read world_size & rank from torch.distributed
    sampler = DistributedSampler(
        index_dataset,
        shuffle=shuffle,
        drop_last=False   # or True if you want every rank to see exactly floor(N/world_size)*world_size samples
    )

    # 3) Wrap it in a BatchSampler to produce lists of indices of size batch_size
    batch_sampler = BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=False
    )

    # 4) DataLoader that uses our batch_sampler + direct get_batch()
    return DataLoader(
        index_dataset,
        batch_sampler=batch_sampler,
        collate_fn=lambda indices: dataset.get_batch(indices),
        num_workers=0,
        **{k: v for k, v in kwargs.items() if k not in ['batch_size', 'shuffle']}
    )