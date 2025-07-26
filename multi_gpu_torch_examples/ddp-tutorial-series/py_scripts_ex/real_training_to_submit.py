import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from deep_learning.NN_datasets import NoNoiseDataset
from deep_learning.NN_datasets.dataloaders import distributed_dataloader
from deep_learning.NN_models import ResNet50
import torch.nn.functional as F
from dataclasses import dataclass
from noise_applicator.noisers.base_noiser import BaseNoiser
from abc import ABC, abstractmethod
from typing import Callable, Any, Union, Optional, List
from pathlib import Path
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

class ImgLastProc(ABC):
    @abstractmethod
    def __call__(self, imgs : torch.Tensor)-> torch.Tensor:
        # here we have imgs bacause we are intedned to always work with batches at this stage
        pass

class Normalizer(ImgLastProc):
    def __call__(self, imgs : torch.Tensor)-> torch.Tensor:
        max_val = imgs.amax(dim=(2, 3), keepdim=True)
        imgs = imgs / max_val

        return imgs

@dataclass
class InputData:
    catalog_name_train : str
    catalog_name_test  : str
    samples_used : int
    samples_used_test : int

    img_size     : int
    img_width    : float

    upscaling    : int

    noiser       : BaseNoiser
    last_image_proc : ImgLastProc

@dataclass
class ChoosenModel:
    model : Callable[..., nn.Module]
    model_init : dict
 #   checkpoint : Optional [Path] = None  # this for now is not handled here

@dataclass
class TrainingSettings:
    optimizer : torch.optim
    # the scheduler is hard coded
    batch_size : int
    test_every_n_batches : int
    compile_model : bool


@dataclass 
class TrainerConfig:
    stages : List[InputData]
    choosen_model : ChoosenModel
    training_settings : TrainingSettings






class Trainer:
    '''
        From this class I want: 
            - To be able to define the the schedule of training,
            - To log stuff to weights and biases,
            - To log checkpoints appropriately
            - To log all the setting of a run, so that it is repetible
            - To be able to start a traininig run from previous checkpoints

    '''

    def __init__(
            self,
            train_config : TrainerConfig,
            local_rank : int,
            gpu_id : int,

    ):
        self.local_rank = local_rank
        self.train_config = train_config
        self.gpu_id = gpu_id

    def _find_better_method_name(self):
        loss = nn.CrossEntropyLoss()
        optimizer = self.train_config.training_settings.optimizer
        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',        # or 'max' if you’re monitoring e.g. accuracy
            factor=1/10.,        # multiply LR by 0.2 when triggered
            patience=15,       # wait 10 epochs with no improvement
            verbose=True,      # prints a message when LR is reduced
            threshold=1e-3,    # “improvement” threshold (optional)
            threshold_mode='rel'  # relative change (optional)
        )
        self.wb_logging =wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="francescocitterio99-max-planck-society",
            # Set the wandb project where this run will be logged.
            project=self.classifier_name,
            # Track hyperparameters and run metadata.
            config={
                "no_config": "no_config"
            },
        )
        

        return optimizer, scheduler, loss

    def _load_loaders(self, stage : InputData):

        train_dataset = NoNoiseDataset(
            stage.catalog_name_train,
            grid_pixel_side=stage.img_size,
            grid_width_arcsec=stage.img_width, #NOTE : this should not be a stage parameter
            broadcasting=True,
            samples_used=stage.samples_used,
            upscaling=stage.upscaling
        )

        test_dataset = NoNoiseDataset(
            stage.catalog_name_test,
            grid_pixel_side=stage.img_size,
            grid_width_arcsec=stage.img_width, #NOTE : this should not be a stage parameter
            broadcasting=True,
            samples_used=stage.samples_used_test,
            upscaling=stage.upscaling
        )


        train_loader = distributed_dataloader(
            train_dataset,
            batch_size=self.train_config.training_settings.batch_size,
        )

        test_loader = distributed_dataloader(
            test_dataset,
            batch_size=self.train_config.training_settings.batch_size
        )
        return train_loader, test_loader


    def _load_model(self, checkpoint_path: Optional[Path] = None):
        model = self.train_config.choosen_model.model(self.train_config.choosen_model.model_init)
        if checkpoint_path is not None:
            model.load_state_dict() # NOTE : add here the correct loading

        model = model.to(self.local_rank) 
        if self.train_config.training_settings.compile_model: 
                         # Move to the correct GPU first
            model = torch.compile(model)              # Then compile for that GPU
            model = DDP(model, device_ids=[self.local_rank])
        return model

    

    def _train_stage(self, stage: InputData, checkpoint = None):
        

        model = self._load_model(checkpoint_path=checkpoint)
        optimizer, scheduler, loss = self._find_better_method_name()
        train_loader, test_loader = self._load_loaders(stage)

        noiser = stage.noiser
        normalizer = stage.last_image_proc # could even not be just a normalizer

        """
            add all the logging that you like
        """

        def _run_batch(source, targets):
            optimizer.zero_grad()
            output = model(source)
            loss = loss(output, targets)
            loss.backward()
            optimizer.step()

        def _run_test():
            model.eval()
            with torch.no_grad():
                pass
            model.train()

        def _run_epoch(epoch):

            b_sz = len(next(iter(train_loader))[0])
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}") # check this
            #self.train_data.sampler.set_epoch(epoch)
            for index, (source, targets) in enumerate(train_loader):
                # source = source.to(self.gpu_id) # they should already live there
                # targets = targets.to(self.gpu_id)
                source = noiser(source)
                source = normalizer(source)
                
                _run_batch(source, targets)

                if index % self.train_config.training_settings.test_every_n_batches == 0:
                    _run_test()

        return checkpoint
    

    def Train(self):
        checkpoint = None
        for stage in self.train_config.stages:
            checkpoint = self._train_stage(stage, checkpoint)



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    # Switch to Gloo so we never invoke NCCL's P2P queries
    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size
    )

    # Map each rank onto whichever GPU(s) you have
    n_gpus = torch.cuda.device_count()
    device_id = rank % n_gpus
    torch.cuda.set_device(device_id)
    print(f"[rank {rank}] running on cuda:{device_id}")
    return device_id

def cleanup():
    dist.destroy_process_group()


def single_proc(rank, world_size):
    device = setup(rank, world_size)

    # Build your dataset & loader (shards automatically via DistributedSampler)
    dataset = NoNoiseDataset(
        'test2',
        grid_pixel_side=100,
        grid_width_arcsec=8.,
        broadcasting=True,
        samples_used=50
    )
    loader = distributed_dataloader(dataset, batch_size=2)

    # Move model to GPU, wrap in DDP *without* device_ids
    model = ResNet50(num_classes=2).to(device)
    ddp_model = DDP(model)  # Gloo will handle CUDA tensors for all_reduce/broadcast :contentReference[oaicite:0]{index=0}

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)

    # One-epoch example
    for images, targets in loader:
        images  = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = ddp_model(images)
        loss = F.cross_entropy(outputs, targets)
        print(loss)
        loss.backward()
        optimizer.step()

    cleanup()

def main():
    world_size = 2
    mp.spawn(single_proc,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
