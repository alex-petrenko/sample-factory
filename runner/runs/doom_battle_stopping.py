from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [42, 42]),
    ('ppo_epochs', [1, 2, 4]),
    ('early_stopping', ['True', 'False']),
])

_experiment = Experiment(
    'battle_fs4',
    'python -m train_pytorch --env=doom_battle_hybrid --train_for_seconds=360000 --algo=PPO --use_rnn=True --rollout=64 --num_envs=96 --reward_scale=0.5',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_battle_v39_fs4_stopping_v2', experiments=[_experiment], pause_between_experiments=10, use_gpus=6, experiments_per_gpu=2, max_parallel=12)


