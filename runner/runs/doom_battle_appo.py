from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [42, 42]),
])

_experiment_10_10 = Experiment(
    'battle_fs4_10_10',
    'python -m algorithms.appo.train_appo --env=doom_battle_hybrid --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=10 --num_envs_per_worker=10 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048',
    _params.generate_params(randomize=False),
)

_experiment_20_20 = Experiment(
    'battle_fs4_20_20',
    'python -m algorithms.appo.train_appo --env=doom_battle_hybrid --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048',
    _params.generate_params(randomize=False),
)

_experiment_40_20 = Experiment(
    'battle_fs4_40_20',
    'python -m algorithms.appo.train_appo --env=doom_battle_hybrid --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=40 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048',
    _params.generate_params(randomize=False),
)

_experiment_80_20 = Experiment(
    'battle_fs4_80_20',
    'python -m algorithms.appo.train_appo --env=doom_battle_hybrid --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=80 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048',
    _params.generate_params(randomize=False),
)


RUN_DESCRIPTION = RunDescription('doom_battle_appo_v56_fs4_lag', experiments=[_experiment_10_10, _experiment_20_20, _experiment_40_20, _experiment_80_20], pause_between_experiments=30, use_gpus=2, experiments_per_gpu=1, max_parallel=2)
