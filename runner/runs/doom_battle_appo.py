from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [42, 42]),
    ('adam_beta1', [0.5, 0.9]),
])

_experiment_32 = Experiment(
    'battle_fs4_32',
    'python -m algorithms.appo.train_appo --env=doom_battle_hybrid --train_for_seconds=360000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=1024',
    _params.generate_params(randomize=False),
)

_experiment_16 = Experiment(
    'battle_fs4_16',
    'python -m algorithms.appo.train_appo --env=doom_battle_hybrid --train_for_seconds=360000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=16 --recurrence=16 --macro_batch=1024',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_battle_appo_v47_fs4_speed', experiments=[_experiment_32, _experiment_16], pause_between_experiments=30, use_gpus=4, experiments_per_gpu=2, max_parallel=8)
