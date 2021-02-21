from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('quads_collision_falloff_radius', [4.0]),
    ('quads_collision_reward', [5.0]),
    ('quads_collision_smooth_max_penalty', [10.0]),
    ('quads_neighbor_encoder_type', ['attention']),
    ('replay_buffer_sample_prob', [0.75]),
])

PBT_CLI = QUAD_BASELINE_CLI + (
    ' --pbt_replace_reward_gap=0.2 --pbt_replace_reward_gap_absolute=200.0 --pbt_period_env_steps=10000000 --pbt_start_mutation=50000000 --with_pbt=True --num_policies=8'
    ' --pbt_mix_policies_in_one_env=False'
    ' --num_workers=72 --num_envs_per_worker=10'
)

_experiment = Experiment(
    'quad_mix_baseline-8_pbt',
    PBT_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_baseline_pbt8_v115', experiments=[_experiment])
