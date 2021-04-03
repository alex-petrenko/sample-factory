from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

NUM_WORKERS = 20  # typically num logical cores
NUM_WORKERS_VOXEL_ENV = 10  # typically num logical cores / 2, limited by the num of available Vulkan contexts
TIMEOUT_SECONDS = 180
SAMPLER_GPUS = '0'  # replace with '0 1 2 3 4 5 6 7' for 8-GPU server

_basic_cli = f'python -m sample_factory.run_algorithm --algo=DUMMY_SAMPLER --num_workers={NUM_WORKERS} --num_envs_per_worker=1 --experiment=benchmark --timeout_seconds={TIMEOUT_SECONDS}'

_params_basic_envs = ParamGrid([
    ('env', ['doom_benchmark', 'atari_breakout', 'dmlab_benchmark']),
])

_experiment_basic_envs = Experiment(
    'benchmark_basic_envs',
    _basic_cli,
    _params_basic_envs.generate_params(randomize=False),
)


_voxel_env_cli = f'python -m sample_factory.run_algorithm --algo=DUMMY_SAMPLER --num_workers={NUM_WORKERS_VOXEL_ENV} --num_envs_per_worker=1 --experiment=benchmark --sampler_worker_gpus {SAMPLER_GPUS} --voxel_num_envs_per_instance=64 --voxel_num_agents_per_env=2 --voxel_num_simulation_threads=2 --timeout_seconds={TIMEOUT_SECONDS}'

_params_voxel_env = ParamGrid([
    ('env', ['voxel_env_obstacleshard']),
    ('voxel_use_vulkan', [True, False]),
])

_experiment_voxel_env = Experiment(
    'benchmark_voxel_env',
    _voxel_env_cli,
    _params_voxel_env.generate_params(randomize=False),
)


RUN_DESCRIPTION = RunDescription('voxel_bench_sampling', experiments=[_experiment_basic_envs, _experiment_voxel_env])
