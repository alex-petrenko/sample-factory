from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['MiniGrid-MemoryS7-v0', 'MiniGrid-RedBlueDoors-8x8-v0', 'MiniGrid-MemoryS17Random-v0']),
    ('use_rnn', [True, False]),
    ('mem_size', [4, 0]),
])

_experiment = Experiment(
    'mem_minigrid',
    'python -m train_pytorch --algo=PPO --train_for_env_steps=300000000',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('mem_minigrid_v39', experiments=[_experiment], pause_between_experiments=5, use_gpus=2, experiments_per_gpu=4, max_parallel=8)
