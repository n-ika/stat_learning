#!/bin/bash

#SBATCH --export=ROOT_PATH=/projects/jurovlab/stat_learning ### Export environment variables to job

#SBATCH --partition=memory              ### Partition (like a queue in PBS)
#SBATCH --account=jurovlab          ### Account used for job submission
# #SBATCH --requeue

### NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayMain, %a=arraySub
#SBATCH --job-name=concat_rnn_bce_batch9_uniqueinit      ### Job Name
#SBATCH --output=tmp/%x.out               ### File in which to store job output
#SBATCH --error=tmp/%x.err                ### File in which to store job error messages

#SBATCH --time=1-00:00:00                ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1                        ### Number of nodes needed for the job
#SBATCH --cpus-per-task=4                ### Number of gpus to be launched per Task
#SBATCH --mem-per-cpu=128G

### Load needed modules
module purge
module load miniconda3/20240410
source $(conda info --base)/etc/profile.d/conda.sh  # ensure conda commands work
conda activate statenv

### Run your actual program
# srun -u $(which python) $ROOT_PATH/scripts/main.py 
srun -u $(which python) /projects/jurovlab/stat_learning/scripts/concatenate.py rnn bce_batch9_uniqueinit
conda deactivate