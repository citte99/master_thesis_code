#!/bin/bash -l

##SBATCH --qos=debug
#SBATCH -o ./job_outs/job.out.%j
#SBATCH -e ./job_outs/job.err.%j
#SBATCH -D ./

#SBATCH --job-name=multinode-example
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu


#SBATCH --gres=gpu:a100:4
#SBATCH --nvmps

#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescocitterio99@gmail.com
#SBATCH --time=6:00:00

module purge
module load apptainer/1.3.6  


export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)
export NCCL_DEBUG=INFO



apptainer exec --nv \
  --bind "$(pwd)":/workspace \
  --pwd /workspace \
  nv-pytorch.sif \
  bash -lc '

  export PYTHONPATH=/workspace/src:$PYTHONPATH

  torchrun \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --nproc_per_node=${SLURM_GPUS_ON_NODE} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    to_submit_torchrun.py
'

# apptainer exec --nv \
#   --bind "$(pwd)":/workspace \
#   --pwd /workspace \
#   nv-pytorch.sif \
#   bash -lc '

#     # 2) Tell Python to look at your src/ first
#     export PYTHONPATH=/workspace/src:$PYTHONPATH

#     # 3) Run your DDP script on GPU 0
#     torchrun --standalone --nproc_per_node=1 to_submit_torchrun_jup.py
#    '


# apptainer exec --nv nv-pytorch.sif torchrun \
#   --nnodes $SLURM_NNODES \
#   --nproc_per_node $SLURM_GPUS_ON_NODE \
#   --rdzv_id $SLURM_JOB_ID \
#   --rdzv_backend c10d \
#   --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#   multigpu_torchrun.py 50 10



# apptainer exec --nv nv-pytorch.sif torchrun \
#   --standalone \
#   --nproc_per_node=1 \
#   to_submit_torchrun.py


# apptainer exec --nv \ # does not work because we do not have write permission on the container for the target dir of pip install
#   --bind $(pwd):/workspace   \
#   nv-pytorch.sif         \
#   bash -lc "
#     cd /workspace           &&
#     pip install -e .        &&   # install your package in editable mode
#     torchrun --standalone \
#              --nproc_per_node=1 \
#              to_submit_torchrun.py
#   "



# # directly binds the live enviroment -> missing astrpy and probably everything else
# apptainer exec --nv \
#   --bind $(pwd):/workspace \
#   --pwd /workspace \
#   nv-pytorch.sif \
#   bash -lc "
#     export PYTHONPATH=/workspace/src:\$PYTHONPATH  &&
#     torchrun --standalone --nproc_per_node=1 \
#               to_submit_torchrun.py
#   "



# apptainer exec --nv \
#   --bind $(pwd):/workspace \
#   --bind $HOME/.local/lib/python3.12/site-packages:/usr/local/lib/python3.12/site-packages_host \
#   --pwd /workspace \
#   nv-pytorch.sif \
#   bash -lc "
#     export PYTHONPATH=/workspace/src:/usr/local/lib/python3.12/site-packages_host:\$PYTHONPATH
#     torchrun --standalone --nproc_per_node=1 to_submit_torchrun.py
#   "


# apptainer exec --nv \
#   --bind $(pwd):/workspace \                          # your code
#   --bind $CONDA_PREFIX:/host_env \                    # your entire conda env
#   --bind $HOME/.local:/home/user/.local \             # your --user site‑packages & scripts
#   --pwd /workspace \
#   nv-pytorch.sif \
#   bash -lc "
#     # 1) Prepend host conda's bin to PATH
#     export PATH=/host_env/bin:\$PATH

#     # 2) Tell Python where to find host site‑packages
#     export PYTHONPATH=/host_env/lib/python$(python3 -c 'import sys;print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')/site-packages:/home/user/.local/lib/python$(python3 -c 'import sys;print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')/site-packages

#     # 3) (Optional) if you need host scripts in ~/.local/bin
#     export PATH=/home/user/.local/bin:\$PATH

#     # 4) Run your distributed job
#     torchrun --standalone --nproc_per_node=1 to_submit_torchrun.py
#   "


#   apptainer exec --nv \
#   --bind "$(pwd)":/workspace \          \
#   --bind "$CONDA_PREFIX":/host_env  \   \
#   --bind "$HOME/.local":/home/user/.local \
#   --pwd /workspace                      \
#   nv-pytorch.sif                        \
#   bash -lc "\
#     export PATH=/host_env/bin:\$PATH:/home/user/.local/bin && \
#     export PYTHONPATH=/workspace/src:/host_env/lib/python$PYVER/site-packages:/home/user/.local/lib/python$PYVER/site-packages && \
#     torchrun --standalone --nproc_per_node=1 to_submit_torchrun.py
#   "


#   # 1) Determine your Python minor version once:
# PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

# # 2) Run apptainer with clean binds & one-liner bash:
# apptainer exec --nv \
#   --bind "$(pwd)":/workspace \
#   --bind "$CONDA_PREFIX":/host_env \
#   --bind "$HOME/.local":/home/user/.local \
#   --pwd /workspace \
#   nv-pytorch.sif \
#   bash -lc ' \
#     export PATH=/host_env/bin:$PATH:/home/user/.local/bin; \
#     export PYTHONPATH=/workspace/src:/host_env/lib/python'"$PYVER"'/site-packages:/home/user/.local/lib/python'"$PYVER"'/site-packages; \
#     torchrun --standalone --nproc_per_node=1 to_submit_torchrun.py \
#   '


# # uses the downloaded wheels for the unavailable libraries
# PYVER=$(python3 -c 'import sys; printf="%d.%d"; print(printf % (sys.version_info.major, sys.version_info.minor))')

# apptainer exec --nv \
#   --bind "$(pwd)/wheelhouse_container":/wheelhouse      \
#   --bind "$(pwd)":/workspace                            \
#   --bind "/u/fcitterio/conda-envs/py312":/host_env      \
#   --bind "$HOME/.local":/home/user/.local               \
#   --pwd /workspace                                       \
#   nv-pytorch.sif                                         \
#   bash -lc '                                             \
#     export PATH=/host_env/bin:$PATH:/home/user/.local/bin; \
#     export PYTHONPATH=/workspace/src:/host_env/lib/python'"$PYVER"'/site-packages:/home/user/.local/lib/python'"$PYVER"'/site-packages; \
#     torchrun --standalone --nproc_per_node=1 to_submit_torchrun.py \
#   ' 


# apptainer exec --nv \
#   --bind "$(pwd)/wheelhouse_container":/wheelhouse \
#   --bind "$(pwd)":/workspace \
#   --pwd /workspace \
#   nv-pytorch.sif \
#   bash -lc '
#     # 1) Install everything offline into your user site
#     pip install --user --no-index \
#                 --find-links=/wheelhouse \
#                 astropy wandb h5py shapely

#     # 2) Tell Python to look at your src/ first
#     export PYTHONPATH=/workspace/src:$PYTHONPATH

#     # 3) Run your DDP script on GPU 0
#     torchrun --standalone --nproc_per_node=1 to_submit_torchrun.py
#   '
  
  
  #if no installation is needed
  
