import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from deep_learning.NN_datasets import NoNoiseDataset
from deep_learning.NN_datasets.dataloaders import distributed_dataloader
from deep_learning.NN_models import ResNet50
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
from noise_applicator.noisers.base_noiser import BaseNoiser
from abc import ABC, abstractmethod
from typing import Callable, Any, Union, Optional, List
from pathlib import Path
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from config import TRAINED_CLASSIFIERS_DIR
import json
from datetime import datetime

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

@dataclass #single stage configurations
class InputData:
    catalog_name_train: str
    catalog_name_test: str
    samples_used: int
    samples_used_test: int
    img_size: int
    img_width: float
    upscaling: int
    noiser: BaseNoiser
    last_image_proc: ImgLastProc

    def get_dict(self):
        return {
            "catalog_name_train": self.catalog_name_train,
            "catalog_name_test": self.catalog_name_test,
            "samples_used": self.samples_used,
            "samples_used_test": self.samples_used_test,
            "img_size": self.img_size,
            "img_width": self.img_width,
            "upscaling": self.upscaling,
            "noiser": self.noiser.__class__.__name__,
            "last_image_proc": self.last_image_proc.__class__.__name__,
        }

@dataclass
class ChoosenModel:
    model : Callable[..., nn.Module]
    model_init : dict
 #   checkpoint : Optional [Path] = None  # this for now is not handled here
    def get_dict(self):
        dict = {
            'model' : self.model.__name__,
            'model_init' : self.model_init
        }
        return dict

@dataclass
class TrainingSettings:
    optimizer : torch.optim
    # the scheduler is hard coded
    first_lr : float
    following_lr : float
    patience_lr : int
    patience_stage : int
    max_epochs : int
    batch_size : int
    test_every_n_batches : int
    compile_model : bool
    def get_dict(self):
        return {
            "optimizer": self.optimizer.__class__.__name__,
            "first_lr": self.first_lr,
            "following_lr": self.following_lr,
            "patience_lr": self.patience_lr,
            "patience_stage": self.patience_stage,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "test_every_n_batches": self.test_every_n_batches,
            "compile_model": self.compile_model,
        }



@dataclass 
class TrainerConfig:
    stages : List[InputData]
    choosen_model : ChoosenModel
    training_settings : TrainingSettings



#========================================================================EXAMPLE TRAINING CONFIG ============================================================
from noise_applicator.noisers.base_noiser import EuclidNoiser

first_stage = InputData(
    catalog_name_train="test_catalog",
    catalog_name_test="test_catalog",
    samples_used=10,
    samples_used_test=10,
    img_size=128,
    img_width=8.0,
    upscaling=1,
    noiser=EuclidNoiser(),  # Replace with an actual implementation of BaseNoiser
    last_image_proc=Normalizer()
)

choosen_model = ChoosenModel(
    model=ResNet50,
    model_init={"num_classes": 2}
)

training_settings = TrainingSettings(
    optimizer=torch.optim.SGD,
    first_lr=0.01,
    following_lr=0.001,
    patience_lr=5,
    patience_stage=10,
    max_epochs=50,
    batch_size=2,
    test_every_n_batches=2,
    compile_model=False
)

trainer_config = TrainerConfig(
    stages=[first_stage],
    choosen_model=choosen_model,
    training_settings=training_settings
)



#============================================================================================================================================================



class PatientTracker:
    def __init__(self, patience_lr: int, patience_stage: int):
        if patience_lr > patience_stage:
            raise ValueError("Patience lr cannot be higher than patience stage")
        self.patience_stage = patience_stage
        self.patience_lr = patience_lr
        self.counts_lr = 0
        self.counts_tot = 0
        self.last_loss = float('inf')

    def check_new_loss(self, loss, sme):
        bool_lower_lr = False
        bool_stop_stage = False

        if loss < self.last_loss - sme:
            self.counts_lr = 0
            self.counts_tot = 0
        else:
            self.counts_lr += 1
            self.counts_tot += 1

        if self.counts_lr >= self.patience_lr:
            bool_lower_lr = True
            self.counts_lr = 0  # optional reset

        if self.counts_tot >= self.patience_stage:
            bool_stop_stage = True
            self.counts_tot = 0

        self.last_loss = min(self.last_loss, loss)
        return bool_lower_lr, bool_stop_stage



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
            classifier_name : str,
            train_config : TrainerConfig,
            local_rank : int,
            gpu_id : int,

    ):  
        self.classifier_name = classifier_name
        self.local_rank = local_rank
        self.train_config = train_config
        self.gpu_id = gpu_id

    def _find_better_method_name(self, model, lr):
        loss = nn.CrossEntropyLoss()
        loss_test = nn.CrossEntropyLoss(reduction= 'none')
        optimizer = self.train_config.training_settings.optimizer(model.parameters(), lr)

        # self.wb_logging =wandb.init(
        #     # Set the wandb entity where your project will be logged (generally your team name).
        #     entity="francescocitterio99-max-planck-society",
        #     # Set the wandb project where this run will be logged.
        #     project=self.classifier_name,
        #     # Track hyperparameters and run metadata.
        #     config={
        #         "no_config": "no_config"
        #     },
        # )
        

        return optimizer, loss, loss_test #scheduler, 

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


    def _load_model(self, checkpoint = None): # checkpoint are the parameters
        model = self.train_config.choosen_model.model(**self.train_config.choosen_model.model_init)
        if checkpoint is not None:
            model.load_state_dict(checkpoint) # NOTE : add here the correct loading

        model = model.to(self.gpu_id) 
        model = DDP(model, device_ids=[self.gpu_id])
        
        if self.train_config.training_settings.compile_model: 
                         # Move to the correct GPU first
            model = torch.compile(model)              # Then compile for that GPU
            
        return model

    def _save_stage_conf(self, stage: InputData, index: int):
        if self.local_rank == 0:
            path = Path(TRAINED_CLASSIFIERS_DIR) / self.classifier_name / f"{index}_{stage.catalog_name_train}"
            path.mkdir(exist_ok=False)
            conf = stage.get_dict()
            config_path = path / "config.json"
            with open(config_path, "w") as f:
                json.dump(conf, f, indent=4)


    def _save_classifier_conf(self):
        if self.local_rank == 0:
            path = Path(TRAINED_CLASSIFIERS_DIR) / self.classifier_name
            path.mkdir(exist_ok=False)

            config_dict = {
                'model' : self.train_config.choosen_model.get_dict(),
                'training_settings' : self.train_config.training_settings.get_dict()
            }
            config_path = path / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=4)

    def _save_checkpoint(self, stage : InputData, stage_index, epoch, model, optimizer, loss_test, lr, is_last = False):
        if self.local_rank == 0:
            path = Path(TRAINED_CLASSIFIERS_DIR) / self.classifier_name / f"{stage_index}_{stage.catalog_name_train}" / 'checkpoints'
            path.mkdir(exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test loss': loss_test,
                'learning_rate': lr,
                # Any other training state you want to preserve
            }
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            checkpoint["timestamp"] = timestamp
            if is_last == False:
                torch.save(checkpoint, path / f'checkpoint_epoch_{epoch}_{timestamp}.pth')
            else: 
                torch.save(checkpoint, path / f'LAST_checkpoint_epoch_{epoch}_{timestamp}.pth')




    def _train_stage(self, stage: InputData, index_stage : int , checkpoint = None):
        
        self._save_stage_conf(stage, index=index_stage)
        model = self._load_model(checkpoint)
        if index_stage == 0:
            lr = self.train_config.training_settings.first_lr
        else:
            lr = self.train_config.training_settings.following_lr

        current_lr = {'value': lr}
        optimizer, loss_fn, loss_test_fn = self._find_better_method_name(model, lr) #scheduler, eventually
        train_loader, test_loader = self._load_loaders(stage)

        noiser = stage.noiser
        normalizer = stage.last_image_proc # could even not be just a normalizer

        """
            add all the logging that you like
        """

        

        def _run_batch(source, targets):
            optimizer.zero_grad()
            output = model(source)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()

        def _run_test():
            model.eval()
            losses = []

            with torch.no_grad():
                for source, targets in test_loader:
                    source = noiser(source)
                    source = normalizer(source)
                    output = model(source)

                    # PER‐EXAMPLE losses (unbiased for noise estimate)
                    per_ex = loss_test_fn(output, targets)
                    losses.append(per_ex)  

            # Concatenate to one big 1‑D tensor, shape=(total_examples,)
            losses = torch.cat(losses, dim=0).double()
            N      = losses.numel()

            mean_loss   = losses.mean()
            sample_std  = losses.std(unbiased=True)        # sqrt(1/(N-1) Σ (ℓ - ℓ̄)^2)
            sem         = sample_std / torch.sqrt(torch.tensor(N))        # σ / √N

            model.train()

            return mean_loss, sem
    
        def _run_epoch(epoch, patient_traker : PatientTracker):
            endstage = False
            checkpoint = None
            last_test_loss = None  
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
                    mean_loss_test, sem_test = _run_test()
                    bool_lower_lr, bool_stop_stage = patient_traker.check_new_loss(mean_loss_test, sem_test)
                    if bool_lower_lr:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] /= 10. # Reduce LR after 10 epochs
                            current_lr['value'] /= 10.
                    if bool_stop_stage:
                        self._save_checkpoint( stage, index_stage, epoch, model, optimizer, mean_loss_test, current_lr['value'], is_last = True)
                        endstage = True
                        checkpoint = model.module.state_dict()
                        return endstage, checkpoint, mean_loss_test
                    else:
                        self._save_checkpoint( stage, index_stage, epoch, model, optimizer, mean_loss_test, current_lr['value'], is_last = False)
            return endstage, checkpoint, mean_loss_test

        pat_trak = PatientTracker(self.train_config.training_settings.patience_lr, self.train_config.training_settings.patience_stage)

        endstage = False
        checkpoint = None
        for epoch in range(self.train_config.training_settings.max_epochs):
           print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
           if not endstage:
              endstage, checkpoint, last_test_loss=  _run_epoch(epoch, patient_traker=pat_trak)

        if not endstage:
            # this saves the model if the training went on for all the ephocs
            self._save_checkpoint( stage, index_stage, epoch, model, optimizer, last_test_loss, current_lr['value'], is_last = True)

        return checkpoint
    
    

    def Train(self):
        self._save_classifier_conf()

        checkpoint = None
        for i, stage in enumerate(self.train_config.stages):
            checkpoint = self._train_stage(stage, index_stage=i, checkpoint=checkpoint)



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
    device_id = setup(rank, world_size)

    trainer = Trainer("my_classifier", train_config=trainer_config, local_rank=rank, gpu_id=device_id)
    trainer.Train()
    cleanup()

def main():
    world_size = 1
    mp.spawn(single_proc,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
