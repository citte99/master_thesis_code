#!/bin/bash -l
#SBATCH -o ./job_outs/job.out.%j
#SBATCH -e ./job_outs/job.err.%j
#SBATCH -D ./
#SBATCH --job-name=multinode-example
#SBATCH --nodes=1                    # Actually change to 2!
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpudev              # Use the partition that worked before
#SBATCH --gres=gpu:a100:4
#SBATCH --nvmps
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescocitterio99@gmail.com
#SBATCH --time=00:05:00

module purge
module load apptainer/1.3.6  

# Get master from first node in allocation
export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NCCL_DEBUG=INFO

# Debug output
echo "Starting job on nodes: $SLURM_JOB_NODELIST"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

# Run on all nodes - pass MASTER_ADDR and MASTER_PORT into the container!
srun apptainer exec --nv \
  --bind "$(pwd)":/workspace \
  --pwd /workspace \
  nv-pytorch.sif \
  bash -lc "
  export PYTHONPATH=/workspace/src:\$PYTHONPATH
  export MASTER_ADDR='${MASTER_ADDR}'
  export MASTER_PORT='${MASTER_PORT}'
  echo 'Node starting with MASTER_ADDR='"\$MASTER_ADDR"
  torchrun \
    --nnodes=\${SLURM_NNODES} \
    --node_rank=\${SLURM_NODEID} \
    --nproc_per_node=\${SLURM_GPUS_ON_NODE} \
    --rdzv_id=\${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
    to_submit_torchrun.py
"