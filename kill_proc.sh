#!/bin/bash

echo 'Killing processes...'
kill -9 $(ps aux | grep 'doom-rl/bin/python' | awk '{print $2}')
kill -9 $(ps aux | grep 'vizdoom' | awk '{print $2}')
kill -9 $(ps aux | grep 'train_appo' | awk '{print $2}')

# Ray-related processes
# kill -9 $(ps aux | grep 'ray/core/src' | awk '{print $2}')
# kill -9 $(ps aux | grep 'ray_Actor' | awk '{print $2}')
# kill -9 $(ps aux | grep 'ray_Vector' | awk '{print $2}')
echo 'Done!'