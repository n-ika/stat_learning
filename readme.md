1. sbatch create_data.sbatch   
        creates data, can specify params (i.e., -16/128/... coefficients for -mel spectrogram, or -13/39/... for -mffccs)
2. sbatch main.sbatch
        train models (can specify with which input (i.e., -unigram), which output folder (important!)) -- i.e. /interim or i.e. interim64_log13: 64 for hidden unit size, 13 for coefficients
4. sbatch concatenate.sbatch
        aggregate results from /interim  and save in /results folder (specify both locations, i.e. reading from interim/interim1024_log16/ writing into results/results1024_log16/)
5. sbatch make_statplots.sbatch
        get plots/figures based on results folder pre-specified
        will go into /results/figures (i.e. results/results1024_log13/)