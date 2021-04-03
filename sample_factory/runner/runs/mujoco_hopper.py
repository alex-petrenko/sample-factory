from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('gamma', [0.995]),
    ('gae_lambda', [1.0]),
    ('ppo_clip_value', [10]),
    ('with_vtrace', ['False']),
    ('learning_rate', [0.0001]),
    ('max_grad_norm', [100.0]),
    ('use_rnn', ['False']),
    ('recurrence', [1]),
    ('num_minibatches_to_accumulate', [0]),
    ('device', ['gpu']),
    ('actor_critic_share_weights', ['False']),
    ('max_policy_lag', [1000000]),
    ('adaptive_stddev', ['False']),

    ('ppo_epochs', [20]),
    ('ppo_clip_ratio', [0.3]),
    ('batch_size', [1024]),
    ('num_batches_per_iteration', [10]),
    ('rollout', [128]),
    ('nonlinearity', ['tanh']),
    ('exploration_loss_coeff', [0.0]),
])

_experiment = Experiment(
    'mujoco_hopper',
    'python -m sample_factory.run_algorithm --env=mujoco_hopper --train_for_env_steps=7000000 --algo=APPO --num_workers=16 --num_envs_per_worker=4 --benchmark=False --with_pbt=False',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('mujoco_hopper_v94', experiments=[_experiment])

# python -m runner.run --run=mujoco_halfcheetah_grid_search --runner=processes --max_parallel=8  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
