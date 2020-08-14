from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ppo_epochs', [1]),
])

_experiment = Experiment(
    'quads_pbt',
    'python -m algorithms.appo.train_appo --env=quadrotor_single --train_for_seconds=3600000 --algo=APPO --gamma=0.99 --use_rnn=False --num_workers=72 --num_envs_per_worker=4 --num_policies=8 --ppo_epochs=1 --rollout=128 --recurrence=1 --batch_size=512 --benchmark=False --pbt_replace_reward_gap=0.1 --pbt_replace_reward_gap_absolute=200.0 --pbt_period_env_steps=1000000 --pbt_start_mutation=20000000 --with_pbt=True --adam_eps=1e-8 --nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False --hidden_size=64 --with_vtrace=False --max_policy_lag=100000000 --gae_lambda=1.00 --max_grad_norm=0.0 --ppo_clip_value=5.0 --exploration_loss_coeff=0.00001 --learning_rate=5e-4',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_pbt_v96_v2', experiments=[_experiment])
