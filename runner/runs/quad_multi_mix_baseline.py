from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [1111, 2222, 3333]),
])

_experiment = Experiment(
    'quad_mix_baseline-16_mixed',
    'python -m run_algorithm --env=quadrotor_multi --train_for_env_steps=5000000000 --algo=APPO --use_rnn=False '
    '--num_workers=72 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 '
    '--encoder_custom=quad_multi_encoder_deepset --with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=256 --quads_neighbor_encoder_type=attention --neighbor_obs_type=pos_vel '
    '--quads_settle_reward=0.1 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_collision_smooth_max_penalty=10.0 --quads_collision_vel_penalty_mode=quadratic '
    '--quads_collision_smooth_vel_coeff=2.0 --quads_collision_vel_penalty_radius=6.0 '
    '--quads_collision_smooth_vel_max_penalty=10.0 --quads_num_agents=16',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_mixed_pvg_v112', experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3