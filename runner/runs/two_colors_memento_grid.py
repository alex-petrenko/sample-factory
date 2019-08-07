from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('recurrence', [1, 16]),
    ('memento', [0, 4, 8]),
    ('memento_increment', [0.1, 1.0]),
    ('memento_decrease', [0.1, 1.0]),
])

_experiment = Experiment(
    'two_colors_memento_grid',
    'python -m algorithms.ppo.train_ppo --env=doom_two_colors_easy --train_for_seconds=40000',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('two_colors_memento_grid', experiments=[_experiment], pause_between_experiments=30, use_gpus=6, experiments_per_gpu=2)

run(gridsearch)
