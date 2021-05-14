from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [1111, 2222, 3333]),
])

_experiments = [
    Experiment(
        'battle2_fs4',
        'python -m sample_factory.algorithms.appo.train_appo --env=doom_battle2 --train_for_env_steps=3000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('doom_battle2_appo_v1.119.0_fs4', experiments=_experiments)
