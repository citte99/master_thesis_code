# Datasets/samplers.py
import torch
from torch.utils.data import Sampler

class RandomBatchSampler(Sampler):
    """
    Sampler that returns random batches of indices.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.len_dataset = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(self.len_dataset))

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            indices = torch.randperm(self.len_dataset, generator=generator).tolist()

        for i in range(0, self.len_dataset, self.batch_size):
            yield indices[i:min(i + self.batch_size, self.len_dataset)]

    def __len__(self):
        return (self.len_dataset + self.batch_size - 1) // self.batch_size