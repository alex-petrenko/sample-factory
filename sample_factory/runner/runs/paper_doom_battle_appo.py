from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [1111, 2222, 3333, 4444]),
])

_experiments = [
    Experiment(
        'battle_fs4',
        'python -m sample_factory.algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_envs_per_worker=8 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('paper_doom_battle_appo_v108_fs4', experiments=_experiments)
