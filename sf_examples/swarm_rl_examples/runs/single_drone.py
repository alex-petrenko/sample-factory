from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription
from sf_examples.swarm_rl_examples.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
    ]
)

SMALL_MODEL_CLI = QUAD_BASELINE_CLI + (
    " --train_for_env_steps=10000000000 --rnn_size=16 --neighbor_obs_type=none --quads_local_obs=-1 "
    "--quads_num_agents=1 --replay_buffer_sample_prob=0.0 --anneal_collision_steps=0 --save_milestones_sec=10000 "
    "--quads_neighbor_encoder_type=no_encoder --serial_mode=False --with_wandb=False --wandb_tags sf2"
)

_experiment = Experiment(
    "baseline",
    SMALL_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("quad_single_baseline", experiments=[_experiment])
# python -m sample_factory.runner.run --run=sf_examples.swarm_rl_examples.runs.single_drone --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=4 --num_gpus=1
