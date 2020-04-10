from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('batch_size', [256, 512, 1024]),
    ('ppo_epochs', [1, 2]),
    ('nonlinearity', ['tanh', 'relu']),
    ('learning_rate', [5e-4, 1e-4]),
    ('max_policy_lag', [50]),
])

_experiment = Experiment(
    'quads_gridsearch',
    'run_algorithm --env=quadrotor_single --train_for_env_steps=100000000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=36 --num_envs_per_worker=4 --num_policies=1 --rollout=32 --recurrence=32 --benchmark=False --with_pbt=False --ppo_clip_ratio=0.05 --value_loss_coeff=1',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_gridsearch_v86', experiments=[_experiment])
