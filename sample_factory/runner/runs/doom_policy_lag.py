from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ppo_epochs', [1]),
])

_experiments = [
    Experiment(
        'battle_fs4_100',
        'python -m sample_factory.algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=1000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=10 --num_envs_per_worker=10 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),

    Experiment(
        'battle_fs4_400',
        'python -m sample_factory.algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=1000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),

    Experiment(
        'battle_fs4_800',
        'algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=1000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=40 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('paper_policy_lag_v66_fs4', experiments=_experiments)
