#!/bin/bash

#SBATCH --export=ROOT_PATH=/projects/jurovlab/stat_learning ### Export environment variables to job

#SBATCH --account=jurovlab   ### Account used for job submission
#SBATCH --partition=jurov     #jurov     #jurov ###gpu
#SBATCH --job-name=plot_p    ### Job Name
#SBATCH --time=1-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --output=tmp/%x_%A.out      # %A: job ID, %a: array index; File in which to store job output
#SBATCH --error=tmp/%x_%A.err                ### File in which to store job error messages
#SBATCH --mem=256G                       ### Total Memory for job in MB -- can do K/M/G/T for KB/MB/GB/TB
#SBATCH --cpus-per-task=2                ### Number of cpus/cores to be launched per Task

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

srun -u $(which python) $ROOT_PATH/scripts/make_statplots.py -a=rnn -lt=mse -et onehot -st unigram -lt=mse -ir=/projects/jurovlab/stat_learning/results/
# srun -u $(which python) $ROOT_PATH/scripts/make_statplots.py -a=rnn -lt=mse -et acoustic_new_norm -st unigram -lt=mse -ir=/projects/jurovlab/stat_learning/results_old/ --logy
# srun -u $(which python) $ROOT_PATH/scripts/make_statplots.py -a=rnn -et acoustic_vec_1 -st unigram -lt=mse_acoustic_vec 
# srun -u $(which python) $ROOT_PATH/scripts/make_statplots.py -a=rnn -et acoustic_vec_1 -st unigram -lt=bce_acoustic_vec 

conda deactivate