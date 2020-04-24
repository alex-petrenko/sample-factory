from runner.run_description import RunDescription, Experiment, ParamGrid

_params_earlystop = ParamGrid([
    ('batch_size', [140]),
    ('ppo_epochs', [10, 20]),
    ('num_batches_per_iteration', [20, 50, 100, 200]),
    ('entropy_loss_coeff', [0.0]),
])

_experiment_earlystop = Experiment(
    'quads_gridsearch_long_rollout',
    'run_algorithm --env=quadrotor_single --train_for_env_steps=1000000000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=24 --num_envs_per_worker=2 --num_policies=1 --rollout=700 --recurrence=1 --benchmark=False --with_pbt=False --ppo_clip_ratio=0.05 --batch_size=128 --nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False --hidden_size=64 --with_vtrace=False --max_policy_lag=100000000 --gae_lambda=1.00 --device=cpu',
    _params_earlystop.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_gridsearch_v91_earlystop', experiments=[_experiment_earlystop])
