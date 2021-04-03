from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('recurrence', [32]),
    ('use_rnn', ['True', 'False']),
    ('ppo_epochs', [1, 2, 4]),
])

_experiment = Experiment(
    'doom_sanity',
    'python -m train_pytorch --train_for_env_steps=3000000 --algo=PPO --env=doom_basic',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_sanity_v44', experiments=[_experiment], pause_between_experiments=5, use_gpus=1, experiments_per_gpu=1, max_parallel=1)

