from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['MiniGrid-MemoryS17Random-v0']),
    ('recurrence', [1, 64]),
    ('mem_size', [0, 3]),
])

_experiment = Experiment(
    'minigrid_mem_s17',
    'python -m algorithms.ppo.train_ppo --train_for_seconds=72000 --mem_feature=128 --hidden_size=128 --rollout=128 --num_envs=32',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('minigrid_mem_s17', experiments=[_experiment], pause_between_experiments=5, use_gpus=2, experiments_per_gpu=2)

run(gridsearch)
