#!/bin/bash
python -m algorithms.appo.train_appo --num_policies=1 --num_workers=14 --with_pbt=True --num_envs_per_worker=8 --experiment=quads_multi_pbt --env=quadrotor_multi --algo=APPO
