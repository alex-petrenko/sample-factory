#!/bin/bash
conda activate sf2
cd ~/sample-factory || exit

timeout $TIMEOUT $CMD
if [[ $$? -eq 124 ]]; then
    sbatch $PARTITION--gres=gpu:$GPU -c $CPU --parsable --output $FILENAME-slurm-%j.out $FILENAME
fi
