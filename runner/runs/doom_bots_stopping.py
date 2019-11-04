from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [42, 42]),
    ('ppo_epochs', [1]),
    ('rollout', [64, 128]),
    ('use_rnn', ['True', 'False']),
])

_experiment = Experiment(
    'bots_fs2',
    'python -m train_pytorch --env=doom_dwango5_bots_experimental --train_for_seconds=360000 --algo=PPO --gamma=0.995 --env_frameskip=2 --rollout=64 --num_envs=96 --reward_scale=0.5 --early_stopping=True',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_bots_v43_fs2', experiments=[_experiment], pause_between_experiments=10, use_gpus=4, experiments_per_gpu=2, max_parallel=8)
