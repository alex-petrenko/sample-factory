from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ppo_epochs', [1]),
])

_experiment = Experiment(
    'static_goal-agents_8',
    'python -m run_algorithm --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False --num_workers=72 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 --nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False --hidden_size=256 --with_vtrace=False --max_policy_lag=100000000 --gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 --extend_obs=True --quads_use_numba=True --quads_num_agents=8 --quads_episode_duration=10.0 --quads_mode=static_goal --quads_dist_between_goals=0.0 --quads_collision_reward=1.0 --encoder_custom=quad_multi_encoder --pbt_replace_reward_gap=0.1 --pbt_replace_reward_gap_absolute=200.0 --pbt_period_env_steps=1000000 --pbt_start_mutation=20000000 --with_pbt=True --num_policies=8',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_same_goal_pbt_v112', experiments=[_experiment])

# this is just a placeholder for command line. You can run it through runner or just copy-paste to command line and run
