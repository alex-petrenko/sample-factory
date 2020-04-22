from runner.run_description import RunDescription, Experiment, ParamGrid

common_args = [
    ('batch_size', [256]),
    ('ppo_epochs', [1]),
    ('nonlinearity', ['tanh']),
    ('learning_rate', [1e-4]),
    ('entropy_loss_coeff', [0.0005]),
    ('actor_critic_share_weights', ['False']),
    ('policy_initialization', ['xavier_uniform']),
    ('max_policy_lag', [50]),
    ('adaptive_stddev', ['False']),
    ('initial_stddev', [1.0]),
    ('hidden_size', [64]),
]

_params_gae = ParamGrid([
    ('with_vtrace', [False]),
    ('gae_lambda', [1.00, 0.99, 0.97, 0.95, 0.93]),
    *common_args,
])

_params_vtrace = ParamGrid([
    ('with_vtrace', [True]),
    ('vtrace_c', [1.05, 1.01, 1.00, 0.99, 0.95]),
    *common_args,
])

_experiment_gae = Experiment(
    'quads_gridsearch_gae',
    'run_algorithm --env=quadrotor_single --train_for_env_steps=1000000000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=36 --num_envs_per_worker=2 --num_policies=1 --rollout=32 --recurrence=32 --benchmark=False --with_pbt=False --ppo_clip_ratio=0.05',
    _params_gae.generate_params(randomize=False),
)

_experiment_vtrace = Experiment(
    'quads_gridsearch_vtrace',
    'run_algorithm --env=quadrotor_single --train_for_env_steps=1000000000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=36 --num_envs_per_worker=2 --num_policies=1 --rollout=32 --recurrence=32 --benchmark=False --with_pbt=False --ppo_clip_ratio=0.05',
    _params_vtrace.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_gridsearch_v89_gae', experiments=[_experiment_gae, _experiment_vtrace])
