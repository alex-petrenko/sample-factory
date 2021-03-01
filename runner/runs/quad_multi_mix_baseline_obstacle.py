from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [1111, 2222, 3333, 4444]),
    ('quads_obstacle_mode', ['dynamic']),
    ('quads_obstacle_num', [1]),
    ('quads_obstacle_type', ['sphere']),
    ('quads_obstacle_traj', ['mix']),
    ('quads_collision_obstacle_reward', [5.0]),
    ('quads_obstacle_obs_mode', ['absolute']),
    ('quads_collision_obst_smooth_max_penalty', [10.0]),
    ('quads_obstacle_hidden_size', [256]),
    ('replay_buffer_sample_prob', [0.0]),
    ('quads_obst_penalty_fall_off', [10.0]),
])

_experiment = Experiment(
    'quad_mix_baseline_obst_mix-8a',
    QUAD_BASELINE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_mix_obst_mix_8a_v115', experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_mix_baseline_obstacle_mix --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_mix_baseline_obstacle_mix --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
