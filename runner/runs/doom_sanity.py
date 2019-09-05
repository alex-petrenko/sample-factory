from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['doom_basic']),
    ('recurrence', [1, 32]),
    ('use_rnn', ['False', 'True']),
])

_experiment = Experiment(
    'doom_sanity_v30',
    'python -m train_pytorch --train_for_seconds=240 --algo=PPO --ppo_epochs=4 --recurrence=1 --use_rnn=False',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_sanity_v30', experiments=[_experiment], pause_between_experiments=5, use_gpus=2, experiments_per_gpu=2, max_parallel=4)

