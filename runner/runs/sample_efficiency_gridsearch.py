from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [1111, 2222]),
    ('env', ['doom_defend_the_line']),
    ('num_envs_per_worker', [4, 8, 16]),
    ('batch_size', [2048, 1024, 512]),
])

_experiments = [
    Experiment(
        'basic_envs_fs4',
        'algorithms.appo.train_appo --train_for_env_steps=200000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('sample_efficiency_gridsearch_v96_fs4', experiments=_experiments)
