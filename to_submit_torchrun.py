import os, torch
rank = int(os.environ.get("LOCAL_RANK", 0))
print(f"[rank{rank}] PID={os.getpid()}  cuda_available={torch.cuda.is_available()}  "
      f"device_count={torch.cuda.device_count()}  "
      + (f"name={torch.cuda.get_device_name(rank)}" if torch.cuda.is_available() else ""))

import os
import gc
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
import wandb

from deep_learning.NN_datasets import NoNoiseDataset
from deep_learning.NN_datasets.dataloaders import distributed_dataloader
from deep_learning.NN_models import ResNet50
from noise_applicator.noisers.base_noiser import BaseNoiser, EuclidNoiser
from config import TRAINED_CLASSIFIERS_DIR


# ------------------------ Image post-proc ------------------------
class ImgLastProc:
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class Normalizer(ImgLastProc):
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        max_val = imgs.amax(dim=(2, 3), keepdim=True)
        imgs = imgs / max_val
        return imgs


# ------------------------ Config dataclasses ------------------------
@dataclass
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
    model: Callable[..., nn.Module]
    model_init: dict

    def get_dict(self):
        return {
            "model": self.model.__name__,
            "model_init": self.model_init,
        }

@dataclass
class TrainingSettings:
    optimizer: torch.optim.Optimizer.__class__
    optimizer_args: dict
    first_lr: float
    following_lr: float
    patience_lr: int
    patience_stage: int
    max_epochs: int
    batch_size: int
#    test_every_n_batches: int
    compile_model: bool
    checkpoints_every_m_test: int

    def get_dict(self):
        return {
            "optimizer": self.optimizer.__name__ if hasattr(self.optimizer, "__name__") else str(self.optimizer),
            "first_lr": self.first_lr,
            "following_lr": self.following_lr,
            "patience_lr": self.patience_lr,
            "patience_stage": self.patience_stage,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
 #           "test_every_n_batches": self.test_every_n_batches,
            "compile_model": self.compile_model,
            "checkpoints_every_m_test": self.checkpoints_every_m_test,
        }

@dataclass
class TrainerConfig:
    stages: List[InputData]
    choosen_model: ChoosenModel
    training_settings: TrainingSettings


# ------------------------ Example training config ------------------------
samp_used = 2_000_000
samp_used_test = 100_000
batch_size = 1024 * 1
max_epochs = 100
#test_every_n_batches = 100

first_stage = InputData(
    catalog_name_train="min_mass_10e11",
    catalog_name_test="min_mass_10e11_test",
    samples_used=samp_used,
    samples_used_test=samp_used_test,
    img_size=80,
    img_width=8.0,
    upscaling=5,
    noiser=EuclidNoiser(),
    last_image_proc=Normalizer(),
)

second_stage = InputData(
    catalog_name_train="min_mass_10e10",
    catalog_name_test="min_mass_10e10_test",
    samples_used=samp_used,
    samples_used_test=samp_used_test,
    img_size=80,
    img_width=8.0,
    upscaling=5,
    noiser=EuclidNoiser(),
    last_image_proc=Normalizer(),
)

third_stage = InputData(
    catalog_name_train="min_mass_10e9",
    catalog_name_test="min_mass_10e9_test",
    samples_used=samp_used,
    samples_used_test=samp_used_test,
    img_size=80,
    img_width=8.0,
    upscaling=5,
    noiser=EuclidNoiser(),
    last_image_proc=Normalizer(),
)

fourth_stage = InputData(
    catalog_name_train="min_mass_10e8_6",
    catalog_name_test="min_mass_10e8_6_test",
    samples_used=samp_used,
    samples_used_test=samp_used_test,
    img_size=80,
    img_width=8.0,
    upscaling=5,
    noiser=EuclidNoiser(),
    last_image_proc=Normalizer(),
)

choosen_model = ChoosenModel(
    model=ResNet50,
    model_init={"num_classes": 2},
)

training_settings = TrainingSettings(
    optimizer=torch.optim.AdamW,
    optimizer_args={"weight_decay":1e-2, "betas":(0.9, 0.999) },
    first_lr=0.001,
    following_lr=0.0001,
    patience_lr=3, #epochs
    patience_stage=8, #stop
    max_epochs=max_epochs,
    batch_size=batch_size,
#    test_every_n_batches=test_every_n_batches,
    compile_model=False,
    checkpoints_every_m_test=5,
)

trainer_config = TrainerConfig(
    stages=[first_stage, second_stage, third_stage, fourth_stage], #first_stage, second_stage, third_stage,
    choosen_model=choosen_model,
    training_settings=training_settings,
)


# ------------------------ Patience tracker ------------------------
class EpochBasedPatienceTracker:
    def __init__(self, patience_epochs_lr: int = 3, patience_epochs_stop: int = 8, 
                 local_rank: int = 0, min_improvement: float = 1e-4):
        self.patience_epochs_lr = patience_epochs_lr
        self.patience_epochs_stop = patience_epochs_stop
        self.min_improvement = min_improvement
        self.local_rank = local_rank
        
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.current_epoch = 0
        
    def end_epoch(self, avg_epoch_loss: float):
        """Call this at the end of each epoch with average loss"""
        self.current_epoch += 1
        
        lr_trigger = False
        stop_trigger = False
        
        # Check for improvement
        if avg_epoch_loss < (self.best_loss - self.min_improvement):
            self.best_loss = avg_epoch_loss
            self.best_epoch = self.current_epoch
        
        epochs_since_improvement = self.current_epoch - self.best_epoch
        
        # Trigger LR reduction
        if epochs_since_improvement >= self.patience_epochs_lr:
            lr_trigger = True
            
        # Trigger early stopping
        if epochs_since_improvement >= self.patience_epochs_stop:
            stop_trigger = True
            
        if self.local_rank == 0:
            wandb.log({
                "patience/epochs_since_improvement": epochs_since_improvement,
                "patience/best_loss": self.best_loss,
                "patience/best_epoch": self.best_epoch,
                "patience/current_epoch": self.current_epoch,
                "patience/lr_trigger": lr_trigger,
                "patience/stop_trigger": stop_trigger,
            })
            
        return lr_trigger, stop_trigger

# ------------------------ Caching utilities ------------------------
def cache_dataset_distributed(stage: InputData, is_train: bool, local_rank: int, world_size: int, gpu_id: int):
    """Cache dataset using all GPUs for parallel processing"""
    
    catalog_name = stage.catalog_name_train if is_train else stage.catalog_name_test
    samples_used = stage.samples_used if is_train else stage.samples_used_test
    split_name = "train" if is_train else "test"
    
    # Create cache directory
    cache_dir = Path(TRAINED_CLASSIFIERS_DIR) / "cache" / f"{catalog_name}_{samples_used}_{stage.img_size}_{stage.upscaling}"
    if local_rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Synchronize all processes
    dist.barrier()
    
    cache_file = cache_dir / f"{split_name}_rank{local_rank}.pt"
    
    # Check if cache already exists
    if cache_file.exists():
        if local_rank == 0:
            print(f"Cache found for {catalog_name} {split_name}, loading...")
        return torch.load(cache_file, map_location=f'cuda:{gpu_id}')
    
    if local_rank == 0:
        print(f"Creating cache for {catalog_name} {split_name} using {world_size} GPUs...")
    
    # Create dataset using your existing distributed loader
    dataset = NoNoiseDataset(
        catalog_name,
        grid_pixel_side=stage.img_size,
        grid_width_arcsec=stage.img_width,
        broadcasting=True,
        samples_used=samples_used,
        upscaling=stage.upscaling,
    )
    
    # Use your distributed dataloader to get data already on GPU
    temp_loader = distributed_dataloader(dataset, batch_size=256)  # Use smaller batch for caching
    
    # Initialize noiser and normalizer
    noiser = stage.noiser
    noiser.set_device(gpu_id)
    normalizer = stage.last_image_proc
    
    cached_images = []
    cached_targets = []
    
    total_batches = len(temp_loader)
    
    for batch_idx, (images, targets) in enumerate(temp_loader):
        # Apply noise and normalization
        processed_images = normalizer(noiser(images))
        
        # Move to CPU for storage
        cached_images.append(processed_images.cpu())
        cached_targets.append(targets.cpu())
        
        if local_rank == 0 and batch_idx % 10 == 0:
            print(f"Caching progress: {batch_idx}/{total_batches} batches")
    
    # Concatenate all batches
    all_images = torch.cat(cached_images, dim=0)
    all_targets = torch.cat(cached_targets, dim=0)
    
    # Save cache file for this rank
    cache_data = {'images': all_images, 'targets': all_targets}
    torch.save(cache_data, cache_file)
    
    if local_rank == 0:
        print(f"Cache saved for {catalog_name} {split_name}")
    
    # Move back to GPU and return
    return {'images': all_images.to(gpu_id), 'targets': all_targets.to(gpu_id)}

# ------------------------ Trainer ------------------------
class Trainer:
    def __init__(
        self,
        classifier_name: str,
        train_config: TrainerConfig,
        local_rank: int,
        gpu_id: int,
        world_size: int,
    ):
        self.classifier_name = classifier_name
        self.local_rank = local_rank
        self.train_config = train_config
        self.gpu_id = gpu_id
        self.world_size = world_size

        if self.local_rank == 0:
            wandb.init(
                mode="offline",
                project=self.classifier_name,
                name=f"{self.classifier_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": self.train_config.choosen_model.get_dict(),
                    "training_settings": self.train_config.training_settings.get_dict(),
                    "stages": [stage.get_dict() for stage in self.train_config.stages],
                    "total_stages": len(self.train_config.stages),
                    "world_size": world_size,
                    "device_count": torch.cuda.device_count(),
                },
            )

    def _find_better_method_name(self, model: nn.Module, lr: float):
        loss = nn.CrossEntropyLoss()
        loss_test = nn.CrossEntropyLoss(reduction="none")
        optimizer = self.train_config.training_settings.optimizer(
            model.parameters(), lr, **self.train_config.training_settings.optimizer_args
        )
        return optimizer, loss, loss_test

    def _load_cached_loaders(self, stage: InputData):
        """Load cached datasets and create efficient DataLoaders"""
        
        # Cache both train and test datasets using all GPUs
        if self.local_rank == 0:
            print(f"Loading/creating cache for stage: {stage.catalog_name_train}")
        
        train_cache = cache_dataset_distributed(stage, is_train=True, 
                                              local_rank=self.local_rank, 
                                              world_size=self.world_size, 
                                              gpu_id=self.gpu_id)
        
        test_cache = cache_dataset_distributed(stage, is_train=False, 
                                             local_rank=self.local_rank, 
                                             world_size=self.world_size, 
                                             gpu_id=self.gpu_id)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(train_cache['images'], train_cache['targets'])
        test_dataset = TensorDataset(test_cache['images'], test_cache['targets'])
        
        # Create distributed samplers
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, 
            num_replicas=self.world_size, 
            rank=self.local_rank,
            shuffle=True
        )
        
        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset, 
            num_replicas=self.world_size, 
            rank=self.local_rank,
            shuffle=False
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.train_config.training_settings.batch_size,
            sampler=train_sampler,
            num_workers=0,  # Data already on GPU
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.train_config.training_settings.batch_size,
            sampler=test_sampler,
            num_workers=0,
            pin_memory=False
        )
        
        return train_loader, test_loader, train_sampler

    def _load_model(self, checkpoint=None):
        model = self.train_config.choosen_model.model(**self.train_config.choosen_model.model_init)
        if checkpoint is not None:
            model.load_state_dict(checkpoint)

        model = model.to(self.gpu_id)
        if self.train_config.training_settings.compile_model:
            model = torch.compile(model)
        model = DDP(model, device_ids=[self.gpu_id])
        return model

    def _save_stage_conf(self, stage: InputData, index: int):
        if self.local_rank == 0:
            path = Path(TRAINED_CLASSIFIERS_DIR) / self.classifier_name / f"{index}_{stage.catalog_name_train}"
            path.mkdir(exist_ok=True)  # idempotent
            with open(path / "config.json", "w") as f:
                json.dump(stage.get_dict(), f, indent=4)

    def _save_classifier_conf(self):
        if self.local_rank == 0:
            path = Path(TRAINED_CLASSIFIERS_DIR) / self.classifier_name
            path.mkdir(exist_ok=True)  # idempotent
            with open(path / "config.json", "w") as f:
                json.dump(
                    {
                        "model": self.train_config.choosen_model.get_dict(),
                        "training_settings": self.train_config.training_settings.get_dict(),
                    },
                    f,
                    indent=4,
                )

    def _save_checkpoint(self, stage: InputData, stage_index, epoch, model, optimizer, loss_test, lr, is_last=False):
        if self.local_rank == 0:
            path = (
                Path(TRAINED_CLASSIFIERS_DIR)
                / self.classifier_name
                / f"{stage_index}_{stage.catalog_name_train}"
                / "checkpoints"
            )
            path.mkdir(exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_loss": float(loss_test),
                "learning_rate": float(lr),
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            }
            name = ("LAST_" if is_last else "") + f"checkpoint_epoch_{epoch}_{checkpoint['timestamp']}.pth"
            torch.save(checkpoint, path / name)

    def _train_stage(self, stage: InputData, index_stage: int, checkpoint=None):
    
        self._save_stage_conf(stage, index=index_stage)
        model = self._load_model(checkpoint)

        lr = self.train_config.training_settings.first_lr if index_stage == 0 else self.train_config.training_settings.following_lr

        if self.local_rank == 0:
            wandb.log(
                {
                    "stage/current_stage": index_stage,
                    "stage/catalog_train": stage.catalog_name_train,
                    "stage/catalog_test": stage.catalog_name_test,
                    "stage/samples_train": stage.samples_used,
                    "stage/samples_test": stage.samples_used_test,
                    "stage/img_size": stage.img_size,
                    "stage/learning_rate_initial": lr,
                }
            )

        optimizer, loss_fn, loss_test_fn = self._find_better_method_name(model, lr)
        train_loader, test_loader, train_sampler = self._load_cached_loaders(stage)

        def _run_batch(source, targets):
            optimizer.zero_grad()
            output = model(source)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            return loss

        def _run_test():
            device = torch.device(f"cuda:{self.gpu_id}")
            model.eval()
            losses = []
            with torch.no_grad():
                for source, targets in test_loader:
                    # Data is already processed and on GPU
                    output = model(source)
                    per_ex = loss_test_fn(output, targets)  # per-example losses
                    losses.append(per_ex)

            losses = torch.cat(losses, dim=0).double()
            total_loss = losses.sum().to(device)
            count = torch.tensor(losses.numel(), device=device, dtype=torch.long)

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

            mean_global = total_loss / count.float()

            if self.local_rank == 0:
                wandb.log(
                    {
                        "test/loss_mean_global": float(mean_global),
                        "test/num_test_samples_global": int(count),
                    }
                )

            model.train()
            return mean_global

        # Initialize epoch-based tracker
        tracker = EpochBasedPatienceTracker(
            patience_epochs_lr=self.train_config.training_settings.patience_lr,
            patience_epochs_stop=self.train_config.training_settings.patience_stage,
            local_rank=self.local_rank,
        )

        # Main epoch loop
        for epoch in range(self.train_config.training_settings.max_epochs):
            device = torch.device(f"cuda:{self.gpu_id}")
            epoch_losses = []
            
            # Set epoch for distributed sampler
            train_sampler.set_epoch(epoch)

            # Show effective batch size
            b_sz = len(next(iter(train_loader))[0])
            if self.local_rank == 0:
                print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}")

            # Log sample images only once at start
            if epoch == 0 and self.local_rank == 0:
                train_batch = next(iter(train_loader))
                test_batch = next(iter(test_loader))
                wandb.log({f"stage_{index_stage}/train_samples": [wandb.Image(train_batch[0][i]) for i in range(min(10, train_batch[0].size(0)))]})
                wandb.log({f"stage_{index_stage}/test_samples": [wandb.Image(test_batch[0][i]) for i in range(min(10, test_batch[0].size(0)))]})

            # Training loop for entire epoch
            for batch_idx, (source, targets) in enumerate(train_loader):
                # Data is already processed and on GPU
                loss = _run_batch(source, targets)
                epoch_losses.append(loss.item())

            # Test once per epoch
            test_loss = _run_test()
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)

            # Check patience using epoch-based tracker
            lr_trigger, stop_trigger = tracker.end_epoch(float(test_loss))

            # Broadcast decisions to all GPUs
            flags = torch.tensor([lr_trigger, stop_trigger], dtype=torch.uint8, device=device)
            if self.local_rank == 0:
                flags[0] = lr_trigger
                flags[1] = stop_trigger
            dist.broadcast(flags, src=0)
            lr_trigger = bool(flags[0].item())
            stop_trigger = bool(flags[1].item())

            # Apply LR reduction
            if lr_trigger:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5

            # Epoch-level logging
            if self.local_rank == 0:
                wandb.log({
                    "epoch/train_loss_avg": avg_train_loss,
                    "epoch/test_loss": float(test_loss),
                    "epoch/learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch/epoch_num": epoch,
                })
                print(f"[GPU {self.gpu_id}] Epoch {epoch}: Train Loss: {avg_train_loss:.6f} | Test Loss: {float(test_loss):.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Save checkpoint periodically or if stopping
            if epoch % 5 == 0 or stop_trigger:
                self._save_checkpoint(stage, index_stage, epoch, model, optimizer, 
                                    test_loss, optimizer.param_groups[0]['lr'], is_last=stop_trigger)

            if stop_trigger:
                if self.local_rank == 0:
                    print(f"[GPU {self.gpu_id}] Early stopping at epoch {epoch}")
                break

        # Final checkpoint if didn't early stop
        if not stop_trigger:
            self._save_checkpoint(stage, index_stage, epoch, model, optimizer, test_loss, optimizer.param_groups[0]['lr'], is_last=True)

        # Cleanup
        final_checkpoint = model.module.state_dict()
        del model, optimizer, train_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()
        return final_checkpoint

    def Train(self):
        self._save_classifier_conf()
        ckpt = None
        for i, stage in enumerate(self.train_config.stages):
            ckpt = self._train_stage(stage, index_stage=i, checkpoint=ckpt)
        wandb.finish()


# ------------------------ DDP setup/cleanup ------------------------
def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return local_rank, world_size

def cleanup():
    dist.destroy_process_group()


# ------------------------ Main ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Automatically set by torchrun")
    args = parser.parse_args()

    local_rank, world_size = setup()

    trainer = Trainer(
        classifier_name="RUN1clusterEpochPatience4",
        train_config=trainer_config,
        local_rank=local_rank,
        gpu_id=local_rank,
        world_size=world_size,
    )
    trainer.Train()

    trainer = Trainer(
        classifier_name="RUN2clusterEpochPatience4",
        train_config=trainer_config,
        local_rank=local_rank,
        gpu_id=local_rank,
        world_size=world_size,
    )
    trainer.Train()

    cleanup()