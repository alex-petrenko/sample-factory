from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('recurrence', [1, 16]),
    ('memento', [0, 4]),
])

_experiment = Experiment(
    'doom_two_colors_memento',
    '--env=doom_two_colors_easy --recurrence=1 --memento=4',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_two_colors_memento', experiments=[_experiment], pause_between_experiments=30, use_gpus=4, experiments_per_gpu=1)

run(gridsearch)
