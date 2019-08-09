from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('recurrence', [1, 16]),
    ('memento_size', [4, 0]),
    ('memento_history', [50, 10, 1]),
])

_experiment = Experiment(
    'mem_multi',
    'python -m algorithms.ppo.train_ppo --env=doom_two_colors_easy',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription(
    'doom_two_colors_memento_multi',
    experiments=[_experiment],
    pause_between_experiments=30, use_gpus=6, experiments_per_gpu=2, max_parallel=12,
)

run(gridsearch)
