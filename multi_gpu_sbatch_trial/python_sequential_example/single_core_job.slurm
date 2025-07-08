#!/bin/bash -l
#
# Single-core example job script for MPCDF Raven.
# In addition to the Python example shown here, the script
# is valid for any single-threaded program, including
# sequential Matlab, Mathematica, Julia, and similar cases.
#
#SBATCH -J PYTHON_SEQ
#SBATCH -o ./out.%j
#SBATCH -e ./err.%j
#SBATCH -D ./
#SBATCH --ntasks=1         # launch job on a single core
#SBATCH --cpus-per-task=1  #   on a shared node
#SBATCH --mem=2000MB       # memory limit for the job
#SBATCH --time=0:10:00

module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2021.05

# Set number of OMP threads to fit the number of available cpus, if applicable.
export OMP_NUM_THREADS=1

# Run single-core program
srun python3 ./python_sequential.py
