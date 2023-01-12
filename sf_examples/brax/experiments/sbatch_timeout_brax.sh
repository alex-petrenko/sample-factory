#!/bin/bash
# replace with your conda environment name
conda activate sf2_39
cd ~/sample-factory || exit

# helps with OOM issues
export XLA_PYTHON_CLIENT_PREALLOCATE=false

timeout $TIMEOUT $CMD
if [[ $$? -eq 124 ]]; then
    sbatch $PARTITION--gres=gpu:$GPU -c $CPU --parsable --output $FILENAME-slurm-%j.out $FILENAME
fi
