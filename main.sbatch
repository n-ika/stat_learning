#!/bin/bash

#SBATCH --export=ROOT_PATH=/projects/jurovlab/stat_learning/ ### Export environment variables to job

#SBATCH --partition=gpu              ### Partition (like a queue in PBS)
#SBATCH --account=jurovlab          ### Account used for job submission
# #SBATCH --requeue
#SBATCH --constraint=gpu-80gb

### NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayMain, %a=arraySub
#SBATCH --job-name=tr_stat_rnn_9       ### Job Name
#SBATCH --output=tmp/%x.out               ### File in which to store job output
#SBATCH --error=tmp/%x.err                ### File in which to store job error messages

#SBATCH --time=1-00:00:00                ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1                        ### Number of nodes needed for the job
#SBATCH --gpus-per-task=1                ### Number of gpus to be launched per Task

### Load needed modules
module purge
module load miniconda3/20240410
source $(conda info --base)/etc/profile.d/conda.sh  # ensure conda commands work
conda activate stat_learning_env

### Run your actual program
# srun -u $(which python) $ROOT_PATH/scripts/main.py 
srun -u $(which python) $ROOT_PATH/scripts/main.py -ma=RNN -hs=8 -lt=bce -btr=9 -o='_batch9'
conda deactivate