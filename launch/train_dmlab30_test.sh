#!/bin/bash
python -m algorithms.appo.train_appo \
--env=dmlab_30 --train_for_seconds=3600000 --algo=APPO \
--gamma=0.99 --use_rnn=True \
--num_workers=20 --num_envs_per_worker=12 \
--num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 \
--benchmark=True \
--dmlab_renderer=software \
--reset_timeout_seconds=300 \
--encoder_custom=dmlab_instructions --encoder_type=resnet --encoder_subtype=resnet_impala --encoder_extra_fc_layers=1 --hidden_size=256 --nonlinearity=relu --rnn_type=lstm \
--max_grad_norm=0.0 \
--dmlab_extended_action_set=True \
--experiment=dmlab_30_test \
--train_for_env_steps=1000000 \
--policy_workers_per_policy=1