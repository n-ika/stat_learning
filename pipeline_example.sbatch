#!/bin/bash

# Setup the needed directories
mkdir -p logs
mkdir -p data
mkdir -p output


# submit First step of pipeline
jobid_create_random=$(sbatch --parsable create_random.sbatch)


# SLURM job dependency syntax
# after:jobid -- after job start or cancellation
# afterany:jobid -- after job was terminated
# afternotok:jobid -- after job terminated in failed state
# afterok:jobid -- after job completes successfully

# afterok:jobid1:jobid2 -- after both jobs succeed
# afterok:jobid1:jobid2,afterany:jobid3 -- after both job1/job2 succeed AND after job3 terminated 
# afterok:jobid1?afternotok:jobid2 -- after job1 succeeds OR job2 terminated 


# submit additional steps dependent on previous step completion
jobid_find_eigh=$(sbatch --parsable --depend=afterok:$jobid_create_random find_eigh.sbatch)
jobid_sum_eigh=$(sbatch --parsable --depend=afterok:$jobid_find_eigh sum_eigh.sbatch)


# submit final job step
sbatch --depend=afterok:$jobid_sum_eigh sort_eigh.sbatch
