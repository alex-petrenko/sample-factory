from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [2222, 3333]),
])

_experiments = [
    Experiment(
        'battle_fs4_400',
        'algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=1000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),

    Experiment(
        'battle_fs4_400_no_vtrace',
        'algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=1000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False --with_vtrace=False',
        _params.generate_params(randomize=False),
    ),

    Experiment(
        'battle_fs4_400_no_ppo',
        'algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=1000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False --with_vtrace=True --ppo_clip_ratio=1000.0 --ppo_clip_value=1000.0',
        _params.generate_params(randomize=False),
    ),

    Experiment(
        'battle_fs4_400_no_correction',
        'algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=1000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False --with_vtrace=False --ppo_clip_ratio=1000.0 --ppo_clip_value=1000.0',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('paper_off_policy_techniques_v66_fs4', experiments=_experiments)
