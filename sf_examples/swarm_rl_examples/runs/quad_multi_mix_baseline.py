from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("quads_collision_reward", [5.0]),
    ]
)

QUAD_BASELINE_CLI = (
    "python -m sf_examples.swarm_rl_examples.train_swarm_rl --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False "
    "--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 "
    "--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform "
    "--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 "
    "--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 "
    "--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 "
    "--encoder_custom=quad_multi_encoder --quads_collision_reward=5.0 "
    "--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel "
    "--quads_settle_reward=0.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 "
    "--quads_local_obs=6 --quads_local_metric=dist "
    "--quads_local_coeff=1.0 --quads_num_agents=8 "
    "--quads_collision_reward=5.0 "
    "--quads_collision_smooth_max_penalty=10.0 "
    "--quads_neighbor_encoder_type=attention "
    "--replay_buffer_sample_prob=0.75 "
    "--anneal_collision_steps=300000000"
)

QUAD_BASELINE_CLI_8 = (
    "python -m sf_examples.swarm_rl_examples.train_swarm_rl --env=quadrotor_multi --train_for_env_steps=2000000000 --algo=APPO --use_rnn=False "
    "--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 "
    "--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform "
    "--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 "
    "--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 "
    "--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 "
    "--encoder_custom=quad_multi_encoder --quads_collision_reward=5.0 "
    "--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel "
    "--quads_settle_reward=0.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 "
    "--quads_local_obs=6 --quads_local_metric=dist "
    "--quads_local_coeff=1.0 --quads_num_agents=8 "
    "--quads_collision_reward=5.0 "
    "--quads_collision_smooth_max_penalty=10.0 "
    "--quads_neighbor_encoder_type=attention "
    "--replay_buffer_sample_prob=0.75 --save_milestones_sec=900 "
    "--anneal_collision_steps=300000000"
)

# For scale, need to change
# num_workers / num_envs_per_worker && quads_num_agents && train_for_env_steps
QUAD_BASELINE_SCALE_CLI_16 = (
    "python -m sf_examples.swarm_rl_examples.train_swarm_rl --env=quadrotor_multi --train_for_env_steps=2000000000 --algo=APPO --use_rnn=False "
    "--num_workers=36 --num_envs_per_worker=2 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 "
    "--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform "
    "--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 "
    "--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 "
    "--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 "
    "--encoder_custom=quad_multi_encoder --quads_collision_reward=5.0 "
    "--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel "
    "--quads_settle_reward=0.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 "
    "--quads_local_obs=6 --quads_local_metric=dist "
    "--quads_local_coeff=1.0 --quads_num_agents=16 "
    "--quads_collision_reward=5.0 "
    "--quads_collision_smooth_max_penalty=10.0 "
    "--quads_neighbor_encoder_type=attention "
    "--replay_buffer_sample_prob=0.75 --save_milestones_sec=900 "
    "--anneal_collision_steps=300000000"
)

# For scale, need to change
# num_workers / num_envs_per_worker && quads_num_agents && train_for_env_steps
QUAD_BASELINE_SCALE_CLI_32 = (
    "python -m sf_examples.swarm_rl_examples.train_swarm_rl --env=quadrotor_multi --train_for_env_steps=2000000000 --algo=APPO --use_rnn=False "
    "--num_workers=18 --num_envs_per_worker=2 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 "
    "--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform "
    "--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 "
    "--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 "
    "--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 "
    "--encoder_custom=quad_multi_encoder --quads_collision_reward=5.0 "
    "--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel "
    "--quads_settle_reward=0.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 "
    "--quads_local_obs=6 --quads_local_metric=dist "
    "--quads_local_coeff=1.0 --quads_num_agents=32 "
    "--quads_collision_reward=5.0 "
    "--quads_collision_smooth_max_penalty=10.0 "
    "--quads_neighbor_encoder_type=attention "
    "--replay_buffer_sample_prob=0.75 --save_milestones_sec=900 "
    "--anneal_collision_steps=300000000"
)

# For scale, need to change
# num_workers / num_envs_per_worker && quads_num_agents && train_for_env_steps
QUAD_BASELINE_SCALE_CLI_48 = (
    "python -m sf_examples.swarm_rl_examples.train_swarm_rl --env=quadrotor_multi --train_for_env_steps=2100000000 --algo=APPO --use_rnn=False "
    "--num_workers=12 --num_envs_per_worker=2 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 "
    "--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform "
    "--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 "
    "--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 "
    "--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 "
    "--encoder_custom=quad_multi_encoder --quads_collision_reward=5.0 "
    "--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel "
    "--quads_settle_reward=0.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 "
    "--quads_local_obs=6 --quads_local_metric=dist "
    "--quads_local_coeff=1.0 --quads_num_agents=48 "
    "--quads_collision_reward=5.0 "
    "--quads_collision_smooth_max_penalty=10.0 "
    "--quads_neighbor_encoder_type=attention "
    "--replay_buffer_sample_prob=0.75 --save_milestones_sec=900 "
    "--anneal_collision_steps=300000000"
)

# For scale, need to change
# num_workers / num_envs_per_worker && quads_num_agents && train_for_env_steps
QUAD_BASELINE_SCALE_CLI_64 = (
    "python -m sf_examples.swarm_rl_examples.train_swarm_rl --env=quadrotor_multi --train_for_env_steps=2100000000 --algo=APPO --use_rnn=False "
    "--num_workers=9 --num_envs_per_worker=2 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 "
    "--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform "
    "--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 "
    "--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 "
    "--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 "
    "--encoder_custom=quad_multi_encoder --quads_collision_reward=5.0 "
    "--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel "
    "--quads_settle_reward=0.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 "
    "--quads_local_obs=6 --quads_local_metric=dist "
    "--quads_local_coeff=1.0 --quads_num_agents=64 "
    "--quads_collision_reward=5.0 "
    "--quads_collision_smooth_max_penalty=10.0 "
    "--quads_neighbor_encoder_type=attention "
    "--replay_buffer_sample_prob=0.75 --save_milestones_sec=900 "
    "--anneal_collision_steps=300000000"
)

# For scale, need to change
# num_workers / num_envs_per_worker && quads_num_agents && train_for_env_steps
QUAD_BASELINE_SCALE_CLI_128 = (
    "python -m sf_examples.swarm_rl_examples.train_swarm_rl --env=quadrotor_multi --train_for_env_steps=2100000000 --algo=APPO --use_rnn=False "
    "--num_workers=4 --num_envs_per_worker=2 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 "
    "--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform "
    "--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 "
    "--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 "
    "--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 "
    "--encoder_custom=quad_multi_encoder --quads_collision_reward=5.0 "
    "--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel "
    "--quads_settle_reward=0.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 "
    "--quads_local_obs=6 --quads_local_metric=dist "
    "--quads_local_coeff=1.0 --quads_num_agents=128 "
    "--quads_collision_reward=5.0 "
    "--quads_collision_smooth_max_penalty=10.0 "
    "--quads_neighbor_encoder_type=attention "
    "--replay_buffer_sample_prob=0.75 --save_milestones_sec=900 "
    "--anneal_collision_steps=300000000"
)

# For scale, need to change
# num_workers / num_envs_per_worker && quads_num_agents && train_for_env_steps
QUAD_BASELINE_SCALE_CLI_256 = (
    "python -m sf_examples.swarm_rl_examples.train_swarm_rl --env=quadrotor_multi --train_for_env_steps=2100000000 --algo=APPO --use_rnn=False "
    "--num_workers=2 --num_envs_per_worker=2 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 "
    "--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform "
    "--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 "
    "--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 "
    "--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 "
    "--encoder_custom=quad_multi_encoder --quads_collision_reward=5.0 "
    "--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel "
    "--quads_settle_reward=0.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 "
    "--quads_local_obs=6 --quads_local_metric=dist "
    "--quads_local_coeff=1.0 --quads_num_agents=256 "
    "--quads_collision_reward=5.0 "
    "--quads_collision_smooth_max_penalty=10.0 "
    "--quads_neighbor_encoder_type=attention "
    "--replay_buffer_sample_prob=0.75 --save_milestones_sec=900 "
    "--anneal_collision_steps=300000000"
)


_experiment = Experiment(
    "quad_mix_baseline-8_mixed",
    QUAD_BASELINE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("quads_multi_mix_baseline_8a_local_v116", experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
