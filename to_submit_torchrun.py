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
    test_every_n_batches: int
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
            "test_every_n_batches": self.test_every_n_batches,
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
samp_used_test = 10_000
batch_size = 1024 * 1
max_epochs = 20
test_every_n_batches = 100

# first_stage = InputData(
#     catalog_name_train="min_mass_10e11",
#     catalog_name_test="min_mass_10e11_test",
#     samples_used=samp_used,
#     samples_used_test=samp_used_test,
#     img_size=80,
#     img_width=8.0,
#     upscaling=5,
#     noiser=EuclidNoiser(),
#     last_image_proc=Normalizer(),
# )

# second_stage = InputData(
#     catalog_name_train="min_mass_10e10",
#     catalog_name_test="min_mass_10e10_test",
#     samples_used=samp_used,
#     samples_used_test=samp_used_test,
#     img_size=80,
#     img_width=8.0,
#     upscaling=5,
#     noiser=EuclidNoiser(),
#     last_image_proc=Normalizer(),
# )

# third_stage = InputData(
#     catalog_name_train="min_mass_10e9",
#     catalog_name_test="min_mass_10e9_test",
#     samples_used=samp_used,
#     samples_used_test=samp_used_test,
#     img_size=80,
#     img_width=8.0,
#     upscaling=5,
#     noiser=EuclidNoiser(),
#     last_image_proc=Normalizer(),
# )

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
    optimizer=torch.optim.Adam,
    optimizer_args={"weight_decay": 1e-4},
    first_lr=0.001,
    following_lr=0.0001,
    patience_lr=2,
    patience_stage=6,
    max_epochs=max_epochs,
    batch_size=batch_size,
    test_every_n_batches=test_every_n_batches,
    compile_model=False,
    checkpoints_every_m_test=5,
)

trainer_config = TrainerConfig(
    stages=[ fourth_stage], #first_stage, second_stage, third_stage,
    choosen_model=choosen_model,
    training_settings=training_settings,
)


# ------------------------ Patience tracker ------------------------
class PatienceTracker:
    def __init__(self, patience_lr: int, patience_stage: int, local_rank: int):
        if patience_lr > patience_stage:
            raise ValueError("patience_lr cannot be higher than patience_stage")
        self.patience_stage = patience_stage
        self.patience_lr = patience_lr
        self.counts_lr = 0
        self.counts_tot = 0
        self.best_loss = float("inf")
        self.local_rank = local_rank

    def check_new_loss(self, loss: torch.Tensor, sem: torch.Tensor):
        lr_trigger = False
        stop_trigger = False

        # Treat as improvement only if we beat best_loss by more than "sem"
        loss_val = float(loss)
        sem_val = float(sem)

        if loss_val < self.best_loss - sem_val:
            self.best_loss = loss_val
            self.counts_lr = 0
            self.counts_tot = 0
        else:
            self.counts_lr += 1
            self.counts_tot += 1

        if self.counts_lr >= self.patience_lr:
            lr_trigger = True
            self.counts_lr = 0

        if self.counts_tot >= self.patience_stage:
            stop_trigger = True

        if self.local_rank == 0:
            wandb.log(
                {
                    "patience/lr_counter": self.counts_lr,
                    "patience/stage_counter": self.counts_tot,
                    "patience/best_loss": float(self.best_loss),
                    "patience/lr_reduction_triggered": lr_trigger,
                    "patience/stage_stop_triggered": stop_trigger,
                }
            )

        return lr_trigger, stop_trigger


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

    def _load_loaders(self, stage: InputData):
        train_dataset = NoNoiseDataset(
            stage.catalog_name_train,
            grid_pixel_side=stage.img_size,
            grid_width_arcsec=stage.img_width,
            broadcasting=True,
            samples_used=stage.samples_used,
            upscaling=stage.upscaling,
        )
        test_dataset = NoNoiseDataset(
            stage.catalog_name_test,
            grid_pixel_side=stage.img_size,
            grid_width_arcsec=stage.img_width,
            broadcasting=True,
            samples_used=stage.samples_used_test,
            upscaling=stage.upscaling,
        )

        train_loader = distributed_dataloader(
            train_dataset, batch_size=self.train_config.training_settings.batch_size
        )
        test_loader = distributed_dataloader(
            test_dataset, batch_size=self.train_config.training_settings.batch_size
        )
        return train_loader, test_loader

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
        current_lr = {"value": lr}

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
        train_loader, test_loader = self._load_loaders(stage)

        noiser = stage.noiser
        noiser.set_device(self.gpu_id)
        normalizer = stage.last_image_proc

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
                    source = noiser(source)
                    source = normalizer(source)
                    output = model(source)
                    per_ex = loss_test_fn(output, targets)  # per-example losses
                    losses.append(per_ex)

            losses = torch.cat(losses, dim=0).double()
            total_loss = losses.sum().to(device)
            sum_sq = (losses**2).sum().to(device)
            count = torch.tensor(losses.numel(), device=device, dtype=torch.long)

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

            mean_global = total_loss / count.float()
            var_global = (sum_sq - count * mean_global**2) / (count - 1).float()
            sem_global = torch.sqrt(var_global / count.float())

            if self.local_rank == 0:
                wandb.log(
                    {
                        "test/loss_mean_global": float(mean_global),
                        "test/loss_sem_global": float(sem_global),
                        "test/num_test_samples_global": int(count),
                    }
                )

            model.train()
            return mean_global, sem_global

        def _run_epoch(epoch: int, tracker: PatienceTracker):
            device = torch.device(f"cuda:{self.gpu_id}")
            endstage = False
            ckpt = None
            last_test_loss = None

            # Show effective batch size of the loader
            b_sz = len(next(iter(train_loader))[0])
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}")

            if epoch == 0 and self.local_rank == 0:
                train_batch = next(iter(train_loader))
                test_batch = next(iter(test_loader))
                train_processed = normalizer(noiser(train_batch[0][:10]))
                test_processed = normalizer(noiser(test_batch[0][:10]))
                wandb.log({f"stage_{index_stage}/train_samples": [wandb.Image(train_processed[i]) for i in range(train_processed.size(0))]})
                wandb.log({f"stage_{index_stage}/test_samples": [wandb.Image(test_processed[i]) for i in range(test_processed.size(0))]})

            for index, (source, targets) in enumerate(train_loader):
                torch.cuda.reset_peak_memory_stats(self.gpu_id)
                start_time = time.time()

                source = noiser(source)
                source = normalizer(source)
                loss = _run_batch(source, targets)

                if index % self.train_config.training_settings.test_every_n_batches == 0:
                    batch_processing_time = time.time() - start_time

                    mean_global, sem_global = _run_test()

                    if self.local_rank == 0:
                        torch.cuda.synchronize(self.gpu_id)
                        peak_mem = torch.cuda.max_memory_allocated(self.gpu_id) / 1e9
                        wandb.log(
                            {
                                "train/batch_loss": float(loss.item()),
                                "train/test_loss_global": float(mean_global),
                                "train/learning_rate": float(current_lr["value"]),
                                "train/epoch": epoch,
                                "train/batch_idx": index,
                                "system/gpu_memory_allocated": torch.cuda.memory_allocated(self.gpu_id) / 1e9,
                                "system/gpu_memory_cached": torch.cuda.memory_reserved(self.gpu_id) / 1e9,
                                "system/batch_time": batch_processing_time,
                                "system/gpu_memory_peak_allocated": peak_mem,
                            }
                        )
                        print(
                            f"[GPU {self.gpu_id} | Epoch {epoch} | Batch {index}] "
                            f"Loss: {loss.item():.4f} | LR: {current_lr['value']:.2e} | "
                            f"Test Loss (global): {float(mean_global):.4f} | "
                            f"Time: {batch_processing_time:.3f}s | "
                            f"Mem Alloc: {torch.cuda.memory_allocated(self.gpu_id) / 1e9:.2f} GB | "
                            f"Peak Mem: {peak_mem:.2f} GB | "
                            f"Mem Cached: {torch.cuda.memory_reserved(self.gpu_id) / 1e9:.2f} GB"
                        )

                    # ---- Global patience decision ----
                    if self.local_rank == 0:
                        lr_trigger, stop_trigger = tracker.check_new_loss(mean_global, sem_global)
                    else:
                        lr_trigger, stop_trigger = False, False

                    flags = torch.tensor([lr_trigger, stop_trigger], dtype=torch.uint8, device=device)
                    dist.broadcast(flags, src=0)
                    lr_trigger = bool(flags[0].item())
                    stop_trigger = bool(flags[1].item())

                    if lr_trigger:
                        for pg in optimizer.param_groups:
                            pg["lr"] = float(pg["lr"]) / 10.0
                        current_lr["value"] /= 10.0

                    if stop_trigger:
                        self._save_checkpoint(stage, index_stage, epoch, model, optimizer, mean_global, current_lr["value"], is_last=True)
                        endstage = True
                        ckpt = model.module.state_dict()
                        return endstage, ckpt, mean_global

                    # periodic checkpointing
                    if (index // self.train_config.training_settings.test_every_n_batches) % self.train_config.training_settings.checkpoints_every_m_test == 0:
                        self._save_checkpoint(stage, index_stage, epoch, model, optimizer, mean_global, current_lr["value"], is_last=False)

            return endstage, ckpt, mean_global

        tracker = PatienceTracker(
            self.train_config.training_settings.patience_lr,
            self.train_config.training_settings.patience_stage,
            local_rank=self.local_rank,
        )

        endstage = False
        ckpt = None
        last_test_loss = None
        for epoch in range(self.train_config.training_settings.max_epochs):
            if not endstage:
                endstage, ckpt, last_test_loss = _run_epoch(epoch, tracker=tracker)

        if not endstage:
            self._save_checkpoint(stage, index_stage, epoch, model, optimizer, last_test_loss, current_lr["value"], is_last=True)

        del model, optimizer, train_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()
        return ckpt

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
        classifier_name="RUN1",
        train_config=trainer_config,
        local_rank=local_rank,
        gpu_id=local_rank,
        world_size=world_size,
    )
    trainer.Train()

    trainer = Trainer(
        classifier_name="RUN2",
        train_config=trainer_config,
        local_rank=local_rank,
        gpu_id=local_rank,
        world_size=world_size,
    )
    trainer.Train()

    cleanup()

# import os

# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

# from deep_learning.NN_datasets import NoNoiseDataset
# from deep_learning.NN_datasets.dataloaders import distributed_dataloader
# from deep_learning.NN_models import ResNet50
# import torch.nn.functional as F
# from dataclasses import dataclass, field, asdict
# from noise_applicator.noisers.base_noiser import BaseNoiser
# from abc import ABC, abstractmethod
# from typing import Callable, Any, Union, Optional, List
# from pathlib import Path
# import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import wandb
# from config import TRAINED_CLASSIFIERS_DIR
# import json
# from datetime import datetime
# import argparse
# import gc
# import time



# class ImgLastProc(ABC):
#     @abstractmethod
#     def __call__(self, imgs : torch.Tensor)-> torch.Tensor:
#         # here we have imgs bacause we are intedned to always work with batches at this stage
#         pass

# class Normalizer(ImgLastProc):
#     def __call__(self, imgs : torch.Tensor)-> torch.Tensor:
#         max_val = imgs.amax(dim=(2, 3), keepdim=True)
#         imgs = imgs / max_val

#         return imgs

# @dataclass #single stage configurations
# class InputData:
#     catalog_name_train: str
#     catalog_name_test: str
#     samples_used: int
#     samples_used_test: int
#     img_size: int
#     img_width: float
#     upscaling: int
#     noiser: BaseNoiser
#     last_image_proc: ImgLastProc

#     def get_dict(self):
#         return {
#             "catalog_name_train": self.catalog_name_train,
#             "catalog_name_test": self.catalog_name_test,
#             "samples_used": self.samples_used,
#             "samples_used_test": self.samples_used_test,
#             "img_size": self.img_size,
#             "img_width": self.img_width,
#             "upscaling": self.upscaling,
#             "noiser": self.noiser.__class__.__name__,
#             "last_image_proc": self.last_image_proc.__class__.__name__,
#         }

# @dataclass
# class ChoosenModel:
#     model : Callable[..., nn.Module]
#     model_init : dict
#  #   checkpoint : Optional [Path] = None  # this for now is not handled here
#     def get_dict(self):
#         dict = {
#             'model' : self.model.__name__,
#             'model_init' : self.model_init
#         }
#         return dict

# @dataclass
# class TrainingSettings:
#     optimizer : torch.optim
#     optimizer_args : dict
#     # the scheduler is hard coded
#     first_lr : float
#     following_lr : float
#     patience_lr : int
#     patience_stage : int
#     max_epochs : int
#     batch_size : int
#     test_every_n_batches : int
#     compile_model : bool
#     checkpoints_every_m_test : int
#     def get_dict(self):
#         return {
#             "optimizer": self.optimizer.__class__.__name__,
#             "first_lr": self.first_lr,
#             "following_lr": self.following_lr,
#             "patience_lr": self.patience_lr,
#             "patience_stage": self.patience_stage,
#             "max_epochs": self.max_epochs,
#             "batch_size": self.batch_size,
#             "test_every_n_batches": self.test_every_n_batches,
#             "compile_model": self.compile_model,
#             "checkpoints_every_m_test" : self.checkpoints_every_m_test
#         }



# @dataclass 
# class TrainerConfig:
#     stages : List[InputData]
#     choosen_model : ChoosenModel
#     training_settings : TrainingSettings



# #========================================================================EXAMPLE TRAINING CONFIG ============================================================
# from noise_applicator.noisers.base_noiser import EuclidNoiser

# samp_used = 2000000
# samp_used_test = 4000
# batch_size = 1024*2
# max_epochs = 20
# test_every_n_batches =100

# first_stage = InputData(
#     catalog_name_train="min_mass_10e11",
#     catalog_name_test="min_mass_10e11_test",
#     samples_used=samp_used,
#     samples_used_test=samp_used_test,
#     img_size=80,
#     img_width=8.0,
#     upscaling=5,
#     noiser=EuclidNoiser(),  # Replace with an actual implementation of BaseNoiser
#     last_image_proc=Normalizer()
# )

# second_stage = InputData(
#     catalog_name_train="min_mass_10e10",
#     catalog_name_test="min_mass_10e10_test",
#     samples_used=samp_used,
#     samples_used_test=samp_used_test,
#     img_size=80,
#     img_width=8.0,
#     upscaling=5,
#     noiser=EuclidNoiser(),  # Replace with an actual implementation of BaseNoiser
#     last_image_proc=Normalizer()
# )

# third_stage = InputData(
#     catalog_name_train="min_mass_10e9",
#     catalog_name_test="min_mass_10e9_test",
#     samples_used=samp_used,
#     samples_used_test=samp_used_test,
#     img_size=80,
#     img_width=8.0,
#     upscaling=5,
#     noiser=EuclidNoiser(),  # Replace with an actual implementation of BaseNoiser
#     last_image_proc=Normalizer()
# )

# fourth_stage = InputData(
#     catalog_name_train="min_mass_10e8_6",
#     catalog_name_test="min_mass_10e8_6_test",
#     samples_used=samp_used,
#     samples_used_test=samp_used_test,
#     img_size=80,
#     img_width=8.0,
#     upscaling=5,
#     noiser=EuclidNoiser(),  # Replace with an actual implementation of BaseNoiser
#     last_image_proc=Normalizer()
# )

# choosen_model = ChoosenModel(
#     model=ResNet50,
#     model_init={"num_classes": 2}
# )

# training_settings = TrainingSettings(
#     optimizer=torch.optim.Adam,
#     optimizer_args = {"weight_decay":1e-4},
#     first_lr=0.001,
#     following_lr=0.0001,
#     patience_lr=5,
#     patience_stage=15,
#     max_epochs=max_epochs,
#     batch_size=batch_size,
#     test_every_n_batches=test_every_n_batches,
#     compile_model=False,
#     checkpoints_every_m_test = 10
# )


# trainer_config = TrainerConfig(
#     stages=[first_stage, second_stage, third_stage, fourth_stage],
#     choosen_model=choosen_model,
#     training_settings=training_settings
# )



# #============================================================================================================================================================



# class PatientTracker:
#     def __init__(self, patience_lr: int, patience_stage: int, local_rank):
#         if patience_lr > patience_stage:
#             raise ValueError("patience_lr cannot be higher than patience_stage")
#         self.patience_stage = patience_stage
#         self.patience_lr = patience_lr
#         self.counts_lr = 0
#         self.counts_tot = 0
#         self.best_loss = float('inf')
#         self.local_rank = local_rank

#     def check_new_loss(self, loss, sme):
#         lr_trigger = False
#         stop_trigger = False

#         # Only treat as a "real" improvement if we beat best_loss by more than 'sme'
#         if loss < self.best_loss - sme:
#             self.best_loss = loss
#             self.counts_lr = 0
#             self.counts_tot = 0
#         else:
#             self.counts_lr += 1
#             self.counts_tot += 1

#         # Time to drop LR?
#         if self.counts_lr >= self.patience_lr:
#             lr_trigger = True
#             self.counts_lr = 0

#         # Time to stop?
#         if self.counts_tot >= self.patience_stage:
#             stop_trigger = True

#         if self.local_rank == 0:
#             wandb.log({
#                 "patience/lr_counter": self.counts_lr,
#                 "patience/stage_counter": self.counts_tot,
#                 "patience/best_loss": float(self.best_loss),
#                 "patience/lr_reduction_triggered": lr_trigger,
#                 "patience/stage_stop_triggered": stop_trigger
#             })

#         return lr_trigger, stop_trigger


# class Trainer:
#     '''
#         From this class I want: 
#             - To be able to define the the schedule of training,
#             - To log stuff to weights and biases,
#             - To log checkpoints appropriately
#             - To log all the setting of a run, so that it is repetible
#             - To be able to start a traininig run from previous checkpoints

#     '''

#     def __init__(
#             self,
#             classifier_name : str,
#             train_config : TrainerConfig,
#             local_rank : int,
#             gpu_id : int,
#             world_size : int

#     ):  
#         self.classifier_name = classifier_name
#         self.local_rank = local_rank
#         self.train_config = train_config
#         self.gpu_id = gpu_id
        
#         if self.local_rank == 0:
#             wandb.init(
#                 mode="offline",
#                 project=self.classifier_name,
#                 name=f"{self.classifier_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#                 config={
#                     # Log entire training configuration
#                     "model": self.train_config.choosen_model.get_dict(),
#                     "training_settings": self.train_config.training_settings.get_dict(),
#                     "stages": [stage.get_dict() for stage in self.train_config.stages],
#                     "total_stages": len(self.train_config.stages),
#                     "world_size": world_size,
#                     "device_count": torch.cuda.device_count()
#                 }
#             )

#     def _find_better_method_name(self, model, lr):
#         loss = nn.CrossEntropyLoss()
#         loss_test = nn.CrossEntropyLoss(reduction= 'none')
#         optimizer = self.train_config.training_settings.optimizer(
#             model.parameters(),
#             lr,
#             **self.train_config.training_settings.optimizer_args
#         )

        

#         return optimizer, loss, loss_test #scheduler, 

#     def _load_loaders(self, stage : InputData):

#         train_dataset = NoNoiseDataset(
#             stage.catalog_name_train,
#             grid_pixel_side=stage.img_size,
#             grid_width_arcsec=stage.img_width, #NOTE : this should not be a stage parameter
#             broadcasting=True,
#             samples_used=stage.samples_used,
#             upscaling=stage.upscaling
#         )

#         test_dataset = NoNoiseDataset(
#             stage.catalog_name_test,
#             grid_pixel_side=stage.img_size,
#             grid_width_arcsec=stage.img_width, #NOTE : this should not be a stage parameter
#             broadcasting=True,
#             samples_used=stage.samples_used_test,
#             upscaling=stage.upscaling
#         )


#         train_loader = distributed_dataloader(
#             train_dataset,
#             batch_size=self.train_config.training_settings.batch_size,
#         )

#         test_loader = distributed_dataloader(
#             test_dataset,
#             batch_size=self.train_config.training_settings.batch_size
#         )
#         return train_loader, test_loader


#     def _load_model(self, checkpoint = None): # checkpoint are the parameters
#         model = self.train_config.choosen_model.model(**self.train_config.choosen_model.model_init)
#         if checkpoint is not None:
#             model.load_state_dict(checkpoint) # NOTE : add here the correct loading

#         model = model.to(self.gpu_id) 
        
        
#         if self.train_config.training_settings.compile_model: 
#                          # Move to the correct GPU first
#             model = torch.compile(model)              # Then compile for that GPU
#         model = DDP(model, device_ids=[self.gpu_id])
#         return model

#     def _save_stage_conf(self, stage: InputData, index: int):
#         if self.local_rank == 0:
#             path = Path(TRAINED_CLASSIFIERS_DIR) / self.classifier_name / f"{index}_{stage.catalog_name_train}"
#             path.mkdir(exist_ok=False)
#             conf = stage.get_dict()
#             config_path = path / "config.json"
#             with open(config_path, "w") as f:
#                 json.dump(conf, f, indent=4)


#     def _save_classifier_conf(self):
#         if self.local_rank == 0:
#             path = Path(TRAINED_CLASSIFIERS_DIR) / self.classifier_name
#             path.mkdir(exist_ok=False)

#             config_dict = {
#                 'model' : self.train_config.choosen_model.get_dict(),
#                 'training_settings' : self.train_config.training_settings.get_dict()
#             }
#             config_path = path / "config.json"
#             with open(config_path, "w") as f:
#                 json.dump(config_dict, f, indent=4)

#     def _save_checkpoint(self, stage : InputData, stage_index, epoch, model, optimizer, loss_test, lr, is_last = False):
#         if self.local_rank == 0:
#             path = Path(TRAINED_CLASSIFIERS_DIR) / self.classifier_name / f"{stage_index}_{stage.catalog_name_train}" / 'checkpoints'
#             path.mkdir(exist_ok=True)
#             checkpoint = {
#                 'epoch': epoch,
#                 'model_state_dict': model.module.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'test loss': loss_test,
#                 'learning_rate': lr,
#                 # Any other training state you want to preserve
#             }
#             timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             checkpoint["timestamp"] = timestamp
#             if is_last == False:
#                 torch.save(checkpoint, path / f'checkpoint_epoch_{epoch}_{timestamp}.pth')
#             else: 
#                 torch.save(checkpoint, path / f'LAST_checkpoint_epoch_{epoch}_{timestamp}.pth')




#     def _train_stage(self, stage: InputData, index_stage : int , checkpoint = None):
        
#         self._save_stage_conf(stage, index=index_stage)
#         model = self._load_model(checkpoint)
#         if index_stage == 0:
#             lr = self.train_config.training_settings.first_lr
#         else:
#             lr = self.train_config.training_settings.following_lr

#         current_lr = {'value': lr}
#         if self.local_rank == 0:
#             wandb.log({"stage/current_stage": index_stage})
#             wandb.log({"stage/catalog_train": stage.catalog_name_train})
#             wandb.log({"stage/catalog_test": stage.catalog_name_test})
#             wandb.log({"stage/samples_train": stage.samples_used})
#             wandb.log({"stage/samples_test": stage.samples_used_test})
#             wandb.log({"stage/img_size": stage.img_size})
#             wandb.log({"stage/learning_rate_initial": lr})
            
        
#         optimizer, loss_fn, loss_test_fn = self._find_better_method_name(model, lr) #scheduler, eventually
#         train_loader, test_loader = self._load_loaders(stage)

#         noiser = stage.noiser
#         noiser.set_device(self.gpu_id)
#         normalizer = stage.last_image_proc # could even not be just a normalizer

#         """
#             add all the logging that you like
#         """

        

#         def _run_batch(source, targets):
#             optimizer.zero_grad()
#             output = model(source)
#             loss = loss_fn(output, targets)
#             loss.backward()
#             optimizer.step()
#             return loss

#         def _run_test():
#             model.eval()
#             losses = []

#             with torch.no_grad():
#                 for source, targets in test_loader:
#                     source = noiser(source)
#                     source = normalizer(source)
#                     output = model(source)

#                     # PER‐EXAMPLE losses (unbiased for noise estimate)
#                     per_ex = loss_test_fn(output, targets)
#                     losses.append(per_ex)  

#             # Concatenate to one big 1‑D tensor, shape=(total_examples,)
#             losses = torch.cat(losses, dim=0).double()
#             N      = losses.numel()

#             mean_loss   = losses.mean()
#             sample_std  = losses.std(unbiased=True)        # sqrt(1/(N-1) Σ (ℓ - ℓ̄)^2)
#             sem         = sample_std / torch.sqrt(torch.tensor(N))        # σ / √N
            
#             if self.local_rank == 0:
#                 wandb.log({"test/loss_mean": mean_loss.item()})
#                 wandb.log({"test/loss_std": sample_std.item()})
#                 wandb.log({"test/loss_sem": sem.item()})
#                 wandb.log({"test/loss_confidence_interval_95": 1.96 * sem.item()})
#                 wandb.log({"test/num_test_samples": N})

#             model.train()

#             return mean_loss, sem
    
#         def _run_epoch(epoch, patient_traker : PatientTracker):
#             endstage = False
#             checkpoint = None
#             last_test_loss = None  
#             b_sz = len(next(iter(train_loader))[0])
#             print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}") # check thisif epoch == 0 and self.local_rank == 0:
#             if epoch == 0 and self.local_rank == 0:
#                 # Get first batches
#                 train_batch = next(iter(train_loader))
#                 test_batch = next(iter(test_loader))

#                 # Process train images
#                 train_source = train_batch[0][:10]  # Take first 10 images
#                 train_processed = normalizer(noiser(train_source))

#                 # Process test images  
#                 test_source = test_batch[0][:10]  # Take first 10 images
#                 test_processed = normalizer(noiser(test_source))

#                 # Log processed images (what the model actually sees)
#                 train_imgs = [wandb.Image(train_processed[i]) for i in range(train_processed.size(0))]
#                 test_imgs = [wandb.Image(test_processed[i]) for i in range(test_processed.size(0))]

#                 wandb.log({f"stage_{index_stage}/train_samples": train_imgs})
#                 wandb.log({f"stage_{index_stage}/test_samples": test_imgs})
            
#             for index, (source, targets) in enumerate(train_loader):
#                 torch.cuda.reset_peak_memory_stats(self.gpu_id)
#                 start_time = time.time()
#                 # source = source.to(self.gpu_id) # they should already live there
#                 # targets = targets.to(self.gpu_id)
#                 source = noiser(source)
#                 source = normalizer(source)
                
#                 loss = _run_batch(source, targets) # loss is return only for logging
                
                
#                 if index % self.train_config.training_settings.test_every_n_batches == 0:
#                     batch_processing_time = time.time() - start_time
                    
#                     mean_loss_test, sem_test = _run_test()
                    
#                     if self.local_rank == 0:
#                         torch.cuda.synchronize(self.gpu_id)  # wait for all GPU ops on this device
#                         peak_mem = torch.cuda.max_memory_allocated(self.gpu_id) / 1e9
#                         wandb.log({"train/batch_loss": loss.item()})
#                         wandb.log({"train/test_loss": mean_loss_test})
#                         wandb.log({"train/learning_rate": current_lr['value']})
#                         wandb.log({"train/epoch": epoch})
#                         wandb.log({"train/batch_idx": index})
#                         wandb.log({"system/gpu_memory_allocated": torch.cuda.memory_allocated(self.gpu_id) / 1e9})
#                         wandb.log({"system/gpu_memory_cached": torch.cuda.memory_reserved(self.gpu_id) / 1e9})
#                         wandb.log({"system/batch_time": batch_processing_time})
#                         wandb.log({"system/gpu_memory_peak_allocated": peak_mem})
#                         print(f"[GPU {self.gpu_id} | Epoch {epoch} | Batch {index}] "
#                               f"Loss: {loss.item():.4f} | LR: {current_lr['value']:.2e} | "
#                               f"Test Loss: {mean_loss_test:.4f} | LR: {current_lr['value']:.2e} | "
#                               f"Time: {batch_processing_time:.3f}s | "
#                               f"Mem Alloc: {torch.cuda.memory_allocated(self.gpu_id) / 1e9:.2f} GB | "
#                               f"Peak Mem: {peak_mem :.2f} GB | "
#                               f"Mem Cached: {torch.cuda.memory_reserved(self.gpu_id) / 1e9:.2f} GB")

                        
                    
#                     bool_lower_lr, bool_stop_stage = patient_traker.check_new_loss(mean_loss_test, sem_test)
#                     if bool_lower_lr:
#                         for param_group in optimizer.param_groups:
#                             param_group['lr'] /= 10. # Reduce LR after 10 epochs
#                             current_lr['value'] /= 10.
#                     if bool_stop_stage:
#                         self._save_checkpoint( stage, index_stage, epoch, model, optimizer, mean_loss_test, current_lr['value'], is_last = True)
#                         endstage = True
#                         checkpoint = model.module.state_dict()
#                         return endstage, checkpoint, mean_loss_test
#                     else:
#                         if index//self.train_config.training_settings.test_every_n_batches % self.train_config.training_settings.checkpoints_every_m_test==0:
#                             self._save_checkpoint( stage, index_stage, epoch, model, optimizer, mean_loss_test, current_lr['value'], is_last = False)
#             return endstage, checkpoint, mean_loss_test

#         pat_trak = PatientTracker(self.train_config.training_settings.patience_lr, self.train_config.training_settings.patience_stage, local_rank= self.local_rank)

#         endstage = False
#         checkpoint = None
#         for epoch in range(self.train_config.training_settings.max_epochs):
#            if not endstage:
#               endstage, checkpoint, last_test_loss=  _run_epoch(epoch, patient_traker=pat_trak)

#         if not endstage:
#             # this saves the model if the training went on for all the ephocs
#             self._save_checkpoint( stage, index_stage, epoch, model, optimizer, last_test_loss, current_lr['value'], is_last = True)
        
#         del model, optimizer, train_loader, test_loader
#         torch.cuda.empty_cache()
#         gc.collect()
#         return checkpoint
    
    

#     def Train(self):
#         self._save_classifier_conf()

#         checkpoint = None
#         for i, stage in enumerate(self.train_config.stages):
#             checkpoint = self._train_stage(stage, index_stage=i, checkpoint=checkpoint)
            
#         wandb.finish()


# def setup():
#     # torchrun will supply these in the env:
#     local_rank = int(os.environ["LOCAL_RANK"])
#     world_size = int(os.environ["WORLD_SIZE"])
#     # pick NCCL for GPUs (or Gloo if you really want)
#     dist.init_process_group(backend="nccl", init_method="env://")
#     torch.cuda.set_device(local_rank)
#     return local_rank, world_size

# def cleanup():
#     dist.destroy_process_group()


# def single_proc(rank, world_size):
#     device_id = setup(rank, world_size)

#     trainer = Trainer("RUN1", train_config=trainer_config, local_rank=rank, gpu_id=device_id)
#     trainer.Train()
    
#     trainer = Trainer("RUN2", train_config=trainer_config, local_rank=rank, gpu_id=device_id)
#     trainer.Train()
#     cleanup()



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # torchrun will inject --local_rank
#     parser.add_argument("--local_rank", type=int, help="Automatically set by torchrun")
#     args = parser.parse_args()

#     # initialize distributed; returns the rank we use as local_rank
#     local_rank, world_size = setup()

#     # Now build and launch your trainer
#     trainer = Trainer(
#         classifier_name="RUN1",
#         train_config=trainer_config,
#         local_rank=local_rank,
#         gpu_id=local_rank,
#         world_size = world_size
#     )
#     trainer.Train()
    
#     trainer = Trainer(
#         classifier_name="RUN2",
#         train_config=trainer_config,
#         local_rank=local_rank,
#         gpu_id=local_rank,
#         world_size = world_size
#     )
#     trainer.Train()

#     cleanup()
