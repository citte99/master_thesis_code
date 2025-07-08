#!/bin/bash -l
#
# Multithreading example job script for MPCDF Raven.
# In addition to the Python example shown here, the script
# is valid for any multi-threaded program, including
# Matlab, Mathematica, Julia, and similar cases.
#
#SBATCH -J PYTHON_MT
#SBATCH -o ./out.%j
#SBATCH -e ./err.%j
#SBATCH -D ./
#SBATCH --ntasks=1         # launch job on
#SBATCH --cpus-per-task=8  #   8 cores on a shared node
#SBATCH --mem=16000MB      # memory limit for the job
#SBATCH --time=0:10:00

module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2021.05

# Set number of OMP threads to fit the number of available cpus, if applicable.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python3 ./python_multithreading.py
