from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333, 4444, 5555, 6666, 7777]),
    ('batch_size', [512]),
    ('ppo_epochs', [1]),
    ('nonlinearity', ['tanh']),
    ('learning_rate', [1e-4]),
    ('exploration_loss_coeff', [0.0]),
    ('initial_stddev', [1.0]),
    ('hidden_size', [64]),
    ('adam_eps', [1e-8]),
])

_experiment = Experiment(
    'quads_gridsearch',
    'python -m run_algorithm --env=quadrotor_single --train_for_env_steps=1000000000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=36 --num_envs_per_worker=2 --num_policies=1 --rollout=128 --recurrence=1 --benchmark=False --with_pbt=False --actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --gae_lambda=1.00 --max_grad_norm=0.0 --ppo_clip_value=5.0',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_benchmark', experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# By using sample-factory: v102, training for 220 M framesteps, average the avg_true_reward of 8 seeds,
# Mean: -229. Standard deviation: 10
# By using sample-factory: v102, training for 280 M framesteps, average the avg_true_reward of 8 seeds,
# Mean: -222. Standard deviation: 10
# By using sample-factory: v102, training for 400 M framesteps, average the avg_true_reward of 8 seeds,
# Mean: -218. Standard deviation: 12