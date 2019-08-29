from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('recurrence', [1, 16]),
    ('batch_size', [512, 1024]),
    ('env_frameskip', [2, 4]),
])

_experiment = Experiment(
    'doom_basic_gridsearch',
    'python -m algorithms.ppo.train_ppo --env=doom_basic --train_for_seconds=120',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_basic_gridsearch', experiments=[_experiment], pause_between_experiments=20)

run(gridsearch)
