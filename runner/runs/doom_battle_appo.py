from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [42, 42, 42, 42]),
])

_experiment = Experiment(
    'battle_fs4',
    'python -m algorithms.appo.train_appo --env=doom_battle_hybrid --train_for_seconds=360000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=10 --num_envs_per_worker=40 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=1024',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_battle_appo_v46_fs4_vtrace', experiments=[_experiment], pause_between_experiments=30, use_gpus=4, experiments_per_gpu=1, max_parallel=4)


