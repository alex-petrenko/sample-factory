#!/bin/bash
python -m algorithms.appo.train_appo \
--env=dmlab_30 --train_for_seconds=3600000 --algo=APPO --gamma=0.99 --use_rnn=True --num_workers=20 --num_envs_per_worker=10 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --benchmark=False --ppo_epochs 1 --experiment dmlab_30_resnet \
--dmlab_renderer=software \
--decorrelate_experience_max_seconds=30 \
--reset_timeout_seconds=300 \
--encoder_custom=dmlab_instructions --encoder_type=resnet --encoder_subtype=resnet_impala --fc_after_encoder=False --hidden_size=256 --nonlinearity=relu \
--dmlab_extended_action_set=True
