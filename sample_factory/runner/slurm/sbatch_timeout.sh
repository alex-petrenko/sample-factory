#!/bin/bash
conda activate sf2
cd ~/sf2

timeout 23h $CMD
if [[ $$? -eq 124 ]]; then
    sbatch $PARTITION--gres=gpu:$GPU -c $CPU --parsable --output $FILENAME-slurm-%j.out $FILENAME
fi
