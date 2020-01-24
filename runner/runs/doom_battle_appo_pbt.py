from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [42]),
])

_experiments = [
    Experiment(
        'battle_fs4_pbt',
        'python -m algorithms.appo.train_appo --env=doom_battle_hybrid --train_for_env_steps=50000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=108 --num_envs_per_worker=30 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False --pbt_period_env_steps=5000000',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('doom_battle_appo_v63_fs4_pbt', experiments=_experiments, pause_between_experiments=60, use_gpus=4, experiments_per_gpu=1, max_parallel=4)
