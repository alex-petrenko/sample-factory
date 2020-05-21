from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['doom_defend_the_center']),
    ('num_envs_per_worker', [8, 16]),
    ('batch_size', [2048]),
])

_experiments = [
    Experiment(
        'basic_envs_fs4',
        'algorithms.appo.train_appo --train_for_env_steps=100000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --policy_workers_per_policy=3',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('sample_efficiency_gridsearch_v97_fs4', experiments=_experiments)
