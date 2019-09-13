from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('recurrence', [32]),
    ('use_rnn', ['True', 'False']),
    ('ppo_epochs', [1, 2, 4, 8, 16]),
])

_experiment = Experiment(
    'doom_sanity',
    'python -m train_pytorch --train_for_seconds=240 --algo=PPO --env=doom_basic',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_sanity_v39_stop', experiments=[_experiment], pause_between_experiments=5, use_gpus=2, experiments_per_gpu=2, max_parallel=4)

