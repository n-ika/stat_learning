#!/bin/bash

#SBATCH --export=ROOT_PATH=/projects/jurovlab/stat_learning ### Export environment variables to job

#SBATCH --account=jurovlab   ### Account used for job submission
#SBATCH --partition=jurov
#SBATCH --array=1-2     ### Array range for array jobs - 70 jobs total
#SBATCH --job-name=qi_ab ### acvec_bce     ### Job Name
#SBATCH --time=1-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --output=tmp/%x_%A_%a.out      # %A: job ID, %a: array index; File in which to store job output
#SBATCH --error=tmp/%x_%A_%a.err                ### File in which to store job error messages
#SBATCH --mem=6G                       ### Total Memory for job in MB -- can do K/M/G/T for KB/MB/GB/TB
#SBATCH --cpus-per-task=4                ### Number of cpus/cores to be launched per Task
#SBATCH --gpus-per-task=1

###SBATCH --constraint=gpu-80gb            ### Only use GPUs that satisfy these constraints
### SLURM can even email you when jobs reach certain states:
#SBATCH --mail-type=BEGIN,END,FAIL       ### accepted types are NONE,BEGIN,END,FAIL,REQUEUE,ALL (does all)
#SBATCH --mail-user=jurov@uoregon.edu
 
### Load needed modules
module purge
module load cuda/12.4.1
module load miniconda3/20240410
source $(conda info --base)/etc/profile.d/conda.sh  # ensure conda commands work
conda activate statenv
# nvidia-smi  # check that gpu is visible

### Run your actual program
# srun -u $(which python) $ROOT_PATH/scripts/main.py -ma=RNN -hs=8 -lt=bce -sn=${SLURM_ARRAY_TASK_ID} # -o='_mse'
# srun -u $(which python) $ROOT_PATH/scripts/main.py -ma=RNN -hs=8 -lt=bce -btr=9 -ui=False -sn=${SLURM_ARRAY_TASK_ID} -o='_batch9'
srun -u $(which python) $ROOT_PATH/scripts/main.py -ma=RNN -hs=8 -lt=bce -btr=9 -ui=True -sn=${SLURM_ARRAY_TASK_ID} -o='_batch9_uniqueinit'
conda deactivate