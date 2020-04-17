from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('batch_size', [512]),
    ('ppo_epochs', [1]),
    ('nonlinearity', ['tanh']),
    ('learning_rate', [1e-4]),
    ('entropy_loss_coeff', [0.001]),
    ('actor_critic_share_weigths', ['False']),
    ('policy_initialization', ['xavier_uniform']),
    ('max_policy_lag', [50]),
    ('adaptive_stddev', ['False']),
    ('initial_stddev', [1.0]),
    ('hidden_size', [64]),
])

_experiment = Experiment(
    'quads_pbt',
    'algorithms.appo.train_appo --env=quadrotor_single --train_for_seconds=3600000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=72 --num_envs_per_worker=10 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --benchmark=False --pbt_replace_reward_gap=0.1 --pbt_replace_reward_gap_absolute=20.0 --pbt_period_env_steps=1000000 --pbt_start_mutation=10000000 --pbt_optimize_batch_size=True --with_pbt=True',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_pbt_v89', experiments=[_experiment])
