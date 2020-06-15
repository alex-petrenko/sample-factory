from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([])

_experiments = [
    Experiment(
        'battle_fs4_pbt',
        'python -m algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=300000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False --num_workers=72 --num_envs_per_worker=36 --num_policies=12 --pbt_replace_reward_gap=0.1 --pbt_replace_reward_gap_absolute=5.0 --pbt_period_env_steps=5000000 --pbt_start_mutation=100000000 --reset_timeout_seconds=300 --with_pbt=True',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('doom_battle_appo_pbt_v79_fs4', experiments=_experiments)
