from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['MiniGrid-MemoryS9-v0', 'MiniGrid-MemoryS17Random-v0', 'MiniGrid-RedBlueDoors-8x8-v0']),
    ('recurrence', [64]),
    ('use_rnn', [True, False]),
    ('ppo_epochs', [2, 4]),
])

_experiment = Experiment(
    'mem_minigrid_v20',
    'python -m train_pytorch --algo=PPO --rollout=64 --num_envs=64 --train_for_env_steps=400000000',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('mem_minigrid_v20', experiments=[_experiment], pause_between_experiments=5, use_gpus=2, experiments_per_gpu=4, max_parallel=8)

run(gridsearch)
