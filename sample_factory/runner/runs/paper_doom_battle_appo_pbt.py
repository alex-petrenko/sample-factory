from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([])

_experiments = [
    Experiment(
        'battle_fs4',
        'python -m sample_factory.algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --num_workers=72 --num_envs_per_worker=32 --num_policies=8 --with_pbt=True',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('paper_doom_battle_appo_pbt_v98_fs4', experiments=_experiments)
