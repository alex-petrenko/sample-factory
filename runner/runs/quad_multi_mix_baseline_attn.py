from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('quads_neighbor_encoder_type', ['attention']),
    ('seed', [0000, 1111, 2222, 3333]),
])

_experiment = Experiment(
    'quad_mix_baseline-8_mixed_attn',
    QUAD_BASELINE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('paper_quads_multi_mix_baseline_8a_attn_v115', experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
