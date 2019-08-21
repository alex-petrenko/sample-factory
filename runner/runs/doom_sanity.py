from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['doom_basic']),
    ('mem_size', [0, 2, 4]),
    ('recurrence', [32]),
    ('use_rnn', [True, False]),
])

_experiment = Experiment(
    'doom_sanity_v10',
    'python -m train_pytorch --train_for_seconds=240 --algo=PPO',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_sanity_v10', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)

run(gridsearch)
