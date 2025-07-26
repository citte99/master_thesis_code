#  Ok so we need to set everything up:
#  
import os


# os.environ["NCCL_P2P_DISABLE"] = "1"
# print("remove os.environ[\"NCCL_P2P_DISABLE\"] = \"1\" in real runs")

import torch.multiprocessing as mp
import torch.distributed as dist
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


# def single_process(rank, names):

#     print("ciao "+ names[rank])

# def main(names):
#     nprocs = 3
#     mp.spawn(fn=single_process, args=(names, ), nprocs=nprocs, join=True)

# if __name__ == "__main__":
#     names = [
#         "carlo",
#         "luciano",
#         "giuseppe"
#     ]

#     main(names)


#NOW LETS SEE IF I CAN DISTRIBUTE THE DATALOADER CORRECTLY: IT DOES!
# from deep_learning.NN_datasets import NoNoiseDataset
# from deep_learning.NN_datasets.dataloaders import distributed_dataloader


# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group('nccl', world_size=world_size, rank=rank)
#     #torch.cuda.set_device (rank)

#     # for testing on a single gpu
#     n_gpus = torch.cuda.device_count()
#     # Map ranks [0,1,...] onto [0,...,n_gpus-1]
#     device_id = rank % n_gpus
#     torch.cuda.set_device(device_id)
#     print(f"[rank {rank}] using device {device_id}")

# def single_process(rank, world_size):
#     setup(rank, world_size=world_size)

#     dataset = NoNoiseDataset(
#     'test2',
#     grid_pixel_side=100,
#     grid_width_arcsec=8.,
#     broadcasting=True,
#     )
    
#     loader = distributed_dataloader(dataset, batch_size=2)

#     iterator = iter(loader) # It is my understanding that since loader has a distributed sampler, 
#                             # this iterator will jump accordingly. Maybe I am wrong

#     for i in range(3):
#         img = next(iterator)


#     print('img correctly extracted')
#     dist.destroy_process_group()

# def main():
#     world_size = 2

    


#     mp.spawn(fn=single_process, args=(world_size, ), nprocs=world_size, join=True)

# if __name__ == "__main__":
   

#     main()






"""
    NOW LETS SET UP MULTI PROCESS TRAINING.
    The new stuff we need to do are
    - wrap the model with DDP, (which requires the device id)
    - that's it : to save snapshots, just check the gpu id so that you do not do it multiple times.
"""




# from deep_learning.NN_datasets import NoNoiseDataset
# from deep_learning.NN_datasets.dataloaders import distributed_dataloader
# from deep_learning.NN_models import ResNet18
# import  torch.nn.functional as F



# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group('nccl', world_size=world_size, rank=rank)
#     #print('remember to go back to nccl backend istead of gloo in real runs')


#     #torch.cuda.set_device (rank)

#     # for testing on a single gpu
#     n_gpus = torch.cuda.device_count()
#     # Map ranks [0,1,...] onto [0,...,n_gpus-1]
#     device_id = rank % n_gpus
#     torch.cuda.set_device(device_id)
#     print(f"[rank {rank}] using device {device_id}")
#     return device_id


# def single_proc(rank, worldsize):
#     device_id = setup(rank, worldsize)

#     dataset = NoNoiseDataset(
#         'test2',
#         grid_pixel_side=100,
#         grid_width_arcsec=8.,
#         broadcasting=True,
#         samples_used=50
#         )
    
#     loader = distributed_dataloader(dataset, batch_size=2)

#     model = ResNet18(2)
#     model.to(device_id)

#     model_ddp = DDP(model, device_ids=[device_id])

#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#     #this is just one epoch
#     for source, targets in loader:
#         optimizer.zero_grad()
#         output = model_ddp(source)
#         loss = F.cross_entropy(output, targets)
#         loss.backward()
#         optimizer.step()


#     dist.destroy_process_group()


# def main():
#     world_size = 2
    


#     mp.spawn(single_proc, args=(world_size,), nprocs=world_size, join=True)





# if __name__ == "__main__":
#     os.environ["NCCL_P2P_DISABLE"] = "1"
#     print("remove os.environ[\"NCCL_P2P_DISABLE\"] = \"1\" in real runs")


#     main()

#NOTE : CHAT GPT  version using gloo backend


# play_around.py

# import os
# # Must be set before any torch.distributed / NCCL code loads
# os.environ["NCCL_P2P_DISABLE"] = "1"  

# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

# from deep_learning.NN_datasets import NoNoiseDataset
# from deep_learning.NN_datasets.dataloaders import distributed_dataloader
# from deep_learning.NN_models import ResNet50
# import torch.nn.functional as F

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '12355'
#     # Switch to Gloo so we never invoke NCCL's P2P queries
#     dist.init_process_group(
#         backend='gloo',
#         rank=rank,
#         world_size=world_size
#     )

#     # Map each rank onto whichever GPU(s) you have
#     n_gpus = torch.cuda.device_count()
#     device_id = rank % n_gpus
#     torch.cuda.set_device(device_id)
#     print(f"[rank {rank}] running on cuda:{device_id}")
#     return device_id

# def cleanup():
#     dist.destroy_process_group()

# def single_proc(rank, world_size):
#     device = setup(rank, world_size)

#     # Build your dataset & loader (shards automatically via DistributedSampler)
#     dataset = NoNoiseDataset(
#         'test2',
#         grid_pixel_side=100,
#         grid_width_arcsec=8.,
#         broadcasting=True,
#         samples_used=50
#     )
#     loader = distributed_dataloader(dataset, batch_size=2)

#     # Move model to GPU, wrap in DDP *without* device_ids
#     model = ResNet50(num_classes=2).to(device)
#     ddp_model = DDP(model)  # Gloo will handle CUDA tensors for all_reduce/broadcast :contentReference[oaicite:0]{index=0}

#     optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)

#     # One-epoch example
#     for images, targets in loader:
#         images  = images.to(device)
#         targets = targets.to(device)

#         optimizer.zero_grad()
#         outputs = ddp_model(images)
#         loss = F.cross_entropy(outputs, targets)
#         print(loss)
#         loss.backward()
#         optimizer.step()

#     cleanup()

# def main():
#     world_size = 2
#     mp.spawn(single_proc,
#              args=(world_size,),
#              nprocs=world_size,
#              join=True)

# if __name__ == "__main__":
#     main()



from substructure_classifier.training_stage_multi import Stage
from substructure_classifier.substructure_classifier_development import SubstructureClassifier



example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnet2")

example_config["training_catalog"]="test_catalog"
example_config["validation_like_train_catalog"]="test_catalog"
example_config["dataset_class_str"]="NoNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False
}

example_config["samples_used_for_training"]=100
example_config["samples_used_for_validation"]=10
example_config["batch_size"]=2
example_config["jump_batch_val"]=5
example_config["learning_rate"]=0.001
example_config["epochs"]=2


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from deep_learning.NN_datasets import NoNoiseDataset
from deep_learning.NN_datasets.dataloaders import distributed_dataloader
from deep_learning.NN_models import ResNet50
import torch.nn.functional as F

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
    gpu_id = setup(rank, world_size)

    my_stage=Stage(classifier_instance=my_classifier, rank = rank,gpu_id=gpu_id , config=example_config, device="cuda")
    my_stage.train(train_ready=True, early_stopping=False)()

    cleanup()

def main():
    world_size = 2
    mp.spawn(single_proc,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()

