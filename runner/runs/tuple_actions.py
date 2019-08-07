from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['doom_battle_discrete', 'doom_battle_hybrid']),
    ('recurrence', [1, 16]),
])

_experiment = Experiment(
    'doom_tuple_actions',
    'python -m algorithms.ppo.train_ppo --train_for_seconds=360000',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_tuple_actions', experiments=[_experiment], pause_between_experiments=60)

run(gridsearch)
