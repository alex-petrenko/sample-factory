from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['MiniGrid-MemoryS7-v0', 'MiniGrid-MemoryS11-v0']),
    ('recurrence', [1, 16]),
    ('memento_size', [0, 4]),
])

_experiment = Experiment(
    'minigrid_mem',
    'python -m algorithms.ppo.train_ppo --train_for_seconds=72000',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_two_colors_memento', experiments=[_experiment], pause_between_experiments=30, use_gpus=2, experiments_per_gpu=2)

run(gridsearch)
