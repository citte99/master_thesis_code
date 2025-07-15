#!/bin/bash -l

#SBATCH -o ./job_outs/job.out.%j
#SBATCH -e ./job_outs/job.err.%j
#SBATCH -D ../py_scripts_ex/

#SBATCH --job-name=multinode-example
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpudev

#SBATCH --gres=gpu:a100:4
#SBATCH --nvmps

#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=00:10:00

module purge
module load apptainer/1.3.6  


# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# echo Node IP: $head_node_ip
# export LOGLEVEL=INFO

# srun torchrun \
# --nnodes $SLURM_NNODES \
# --nproc_per_node $SLURM_GPUS_ON_NODE
# # --rdzv_id $RANDOM \
# # --rdzv_backend c10d \
# # --rdzv_endpoint $head_node_ip:29500 \
# multinode_torchrun.py 50 10


export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)
export NCCL_DEBUG=INFO

apptainer exec --nv ../nv-pytorch.sif torchrun \
  --nnodes $SLURM_NNODES \
  --nproc_per_node $SLURM_GPUS_ON_NODE \
  --rdzv_id $SLURM_JOB_ID \
  --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  multigpu_torchrun.py 50 10
