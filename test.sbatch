#!/bin/bash

#SBATCH --export=ROOT_PATH=/projects/jurovlab/stat_learning ### Export environment variables to job

#SBATCH --account=jurovlab   ### Account used for job submission
#SBATCH --partition=gpu       ### Similar to a queue in PBS
#SBATCH --job-name=GPUjob     ### Job Name
#SBATCH --time=0-00:10:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --gpus=1              ### General REServation of gpu:number of gpus
#SBATCH --output=%x-%j.out               ### File in which to store job output
#SBATCH --error=%x-%j.err                ### File in which to store job error messages
#SBATCH --mem=500M                       ### Total Memory for job in MB -- can do K/M/G/T for KB/MB/GB/TB
#SBATCH --cpus-per-task=1                ### Number of cpus/cores to be launched per Task

#SBATCH --constraint=gpu-10gb            ### Only use GPUs that satisfy these constraints
### SLURM can even email you when jobs reach certain states:
#SBATCH --mail-type=BEGIN,END,FAIL       ### accepted types are NONE,BEGIN,END,FAIL,REQUEUE,ALL (does all)
#SBATCH --mail-user=jurov@uoregon.edu

### Load needed modules
module purge
module load cuda/12.4.1
module load miniconda3/20240410
source $(conda info --base)/etc/profile.d/conda.sh  # ensure conda commands work
conda activate gpu_env
module list

### Run your actual program
nvidia-smi  # check that gpu is visible
which python
srun -u $(which python) $ROOT_PATH/scripts/cuda_test.py 
# srun -u $(which python) $ROOT_PATH/scripts/main.py -ma=RNN -hs=8 -lt=bce -btr=9 -o='_batch9'
conda deactivate