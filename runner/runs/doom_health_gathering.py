from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [1111, 2222]),
    ('env', ['doom_health_gathering_supreme']),
    ('with_vtrace', ['True', 'False']),
    ('gae_lambda', [0.95, 1.0]),
    ('entropy_loss', [0.001, 0.003]),
])

_experiments = [
    Experiment(
        'basic_envs_fs4',
        'algorithms.appo.train_appo --train_for_env_steps=300000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=12 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('doom_health_gathering_v96_fs4', experiments=_experiments)
