from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('batch_size', [256]),
    ('ppo_epochs', [1]),
    ('nonlinearity', ['tanh']),
    ('learning_rate', [1e-3, 1e-4]),
    ('entropy_loss_coeff', [0.001]),
    ('actor_critic_share_weigths', ['True', 'False']),
    ('policy_initialization', ['xavier_uniform']),
    ('max_policy_lag', [50]),
    ('adaptive_stddev', ['False', 'True']),
    ('initial_stddev', [0.5]),
    ('hidden_size', [64, 256]),
])

_experiment = Experiment(
    'quads_gridsearch',
    'run_algorithm --env=quadrotor_single --train_for_env_steps=100000000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=36 --num_envs_per_worker=2 --num_policies=1 --rollout=32 --recurrence=32 --benchmark=False --with_pbt=False --ppo_clip_ratio=0.05 --value_loss_coeff=1.0',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_gridsearch_v88_64', experiments=[_experiment])
