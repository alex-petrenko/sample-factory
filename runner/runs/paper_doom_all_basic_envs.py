from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [1111]),
    ('env', ['doom_my_way_home', 'doom_deadly_corridor', 'doom_defend_the_center', 'doom_defend_the_line', 'doom_health_gathering', 'doom_health_gathering_supreme']),
])

_experiments = [
    Experiment(
        'basic_envs_fs4',
        'algorithms.appo.train_appo --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=12 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('paper_doom_basic_envs_appo_v65_fs4', experiments=_experiments)
