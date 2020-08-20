from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ppo_epochs', [1]),
])

_experiment = Experiment(
    'voxel_env_pbt',
    'python -m algorithms.appo.train_appo --env=voxel_env_v8 --train_for_seconds=360000 --algo=APPO --gamma=0.995 --use_rnn=True --num_envs_per_worker=16 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --actor_worker_gpus 0 1 2 3 4 5 6 7 --num_policies=8 --with_pbt=True --max_grad_norm=0.0 --pbt_replace_reward_gap_absolute=0.1 --exploration_loss=symmetric_kl --exploration_loss_coeff=0.001',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('voxel_env_pbt_v105_env_v8_8p', experiments=[_experiment])
