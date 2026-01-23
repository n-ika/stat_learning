#!/bin/bash

#SBATCH --export=ROOT_PATH=/projects/jurovlab/stat_learning/ ### Export environment variables to job

#SBATCH --partition=jurov              ### Partition (like a queue in PBS)
#SBATCH --account=jurovlab               ### Account used for job submission

### NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayMain, %a=arraySub
#SBATCH --job-name=create_data            ### Job Name
#SBATCH --output=tmp/%x.out               ### File in which to store job output
#SBATCH --error=tmp/%x.err                ### File in which to store job error messages
#SBATCH --time=0-12:00:00                ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --ntasks=1                       ### Number of tasks included in the job
#SBATCH --mem-per-cpu=16G                ### Total Memory for job in MB -- can do K/M/G/T for KB/MB/GB/TB
#SBATCH --ntasks-per-node=1              ### Number of tasks to be launched per Node
#SBATCH --cpus-per-task=16                ### Number of cpus/cores to be launched per Task

module purge
module load miniconda3/20240410
source $(conda info --base)/etc/profile.d/conda.sh  # ensure conda commands work
conda activate statenv
### Run your actual program
srun -u $(which python) $ROOT_PATH/scripts/create_data.py --n_mels=39
conda deactivate
