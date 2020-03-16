from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ppo_epochs', [1]),
])

_experiment = Experiment(
    'quads_pbt',
    'algorithms.appo.train_appo --env=quadrotor_single --train_for_seconds=3600000 --algo=APPO --gamma=0.99 --env_frameskip=2 --use_rnn=True --num_workers=72 --num_envs_per_worker=8 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=1024 --batch_size=1024 --benchmark=False --pbt_replace_reward_gap=0.1 --pbt_replace_reward_gap_absolute=5.0 --pbt_period_env_steps=1000000',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_v79', experiments=[_experiment])
