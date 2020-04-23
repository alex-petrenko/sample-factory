from runner.run_description import RunDescription, Experiment, ParamGrid

_params_clip = ParamGrid([
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
    ('quads_clip_input', ['True', 'False']),
])

_experiment_clip = Experiment(
    'quads_gridsearch_clip',
    'run_algorithm --env=quadrotor_single --train_for_env_steps=1000000000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=36 --num_envs_per_worker=2 --num_policies=1 --rollout=32 --recurrence=32 --benchmark=False --with_pbt=False --ppo_clip_ratio=0.05',
    _params_clip.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_gridsearch_v89_clip', experiments=[_experiment_clip])
