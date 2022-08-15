from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription
from sf_examples.swarm_rl_examples.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI

_params = ParamGrid(
    [
        ("seed", [0000]),
    ]
)

SMALL_MODEL_CLI = QUAD_BASELINE_CLI + (
    " --train_for_env_steps=10000000000 --num_workers=8 --num_envs_per_worker=8 "
    "--quads_num_agents=8 --save_milestones_sec=10000 --async_rl=True --num_batches_to_accumulate=8 "
    "--serial_mode=False --batched_sampling=True"
)

# SMALL_MODEL_CLI = QUAD_BASELINE_CLI + (
#     " --train_for_env_steps=10000000000 --neighbor_obs_type=none "
#     "--quads_num_agents=8 --anneal_collision_steps=0 --save_milestones_sec=10000 "
#     "--quads_neighbor_encoder_type=no_encoder --serial_mode=False --quads_mode=static_same_goal --normalize_input=True --normalize_returns=True"
# )

_experiment = Experiment(
    "baseline_multi_drone",
    SMALL_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("quad_multi_baseline", experiments=[_experiment])
# python -m sample_factory.runner.run --run=sf_examples.swarm_rl_examples.runs.multi_drone --runner=processes --max_parallel=1 --pause_between=1 --experiments_per_gpu=4 --num_gpus=1
