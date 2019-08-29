from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('env', ['MiniGrid-RedBlueDoors-8x8-v0']),
    ('recurrence', [1, 16]),
    ('mem_size', [0, 2]),
])

_experiment = Experiment(
    'minigrid_mem_doors_v2',
    'python -m algorithms.ppo.train_ppo --train_for_seconds=72000 --mem_feature=128 --hidden_size=128',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('minigrid_mem_doors_v2', experiments=[_experiment], pause_between_experiments=5, use_gpus=2, experiments_per_gpu=2)

run(gridsearch)
