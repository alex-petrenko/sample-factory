from runner.run_description import RunDescription, Experiment, ParamGrid

_params_clip = ParamGrid([
    ('seed', [1000, 2000]),
    ('batch_size', [128]),
    ('ppo_epochs', [1, 2]),
    ('num_batches_per_iteration', [5, 10, 15]),
])

_experiment_clip = Experiment(
    'quads_gridsearch_long_rollout',
    'run_algorithm --env=quadrotor_single --train_for_env_steps=1000000000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=36 --num_envs_per_worker=2 --num_policies=1 --rollout=640 --recurrence=1 --benchmark=False --with_pbt=False --ppo_clip_ratio=0.03 --batch_size=128 --nonlinearity=tanh --entropy_loss_coeff=0.0005 --actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False --hidden_size=64 --with_vtrace=False --max_policy_lag=1000 --gae_lambda=1.00',
    _params_clip.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_gridsearch_v89_long_rollout', experiments=[_experiment_clip])
