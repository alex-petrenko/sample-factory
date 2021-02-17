from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [0000, 3333]),
    ('quads_obs_repr', ['xyz_vxyz_R_omega', 'xyz_vxyz_R_omega_wall']),
])

_experiment = Experiment(
    'grid_search_add_wall_collision_func-8_mixed',
    'python -m run_algorithm --env=quadrotor_multi --train_for_env_steps=5000000000 --algo=APPO --use_rnn=False --num_workers=72 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 --nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False --hidden_size=256 --quads_neighbor_hidden_size=256 --with_vtrace=False --max_policy_lag=100000000 --gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 --quads_use_numba=True --quads_num_agents=8 --quads_episode_duration=15.0 --quads_mode=mix --quads_formation_size=0.0 --encoder_custom=quad_multi_encoder_deepset --with_pbt=False --quads_neighbor_encoder_type=attention --quads_collision_reward=5.0 --neighbor_obs_type=pos_vel',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_mixed_pvg_v112', experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_mix_obs_add_walls --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_mix_obs_add_walls --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4