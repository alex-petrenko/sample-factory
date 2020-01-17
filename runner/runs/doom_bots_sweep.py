from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [42]),
])

_experiments = [
    Experiment(
        'bots_128_fs2',
        'python -m algorithms.appo.train_appo --env=doom_dwango5_bots_experimental --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --rollout=32 --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048',
        _params.generate_params(randomize=False),
    ),

    Experiment(
        'bots_128_fs4',
        'python -m algorithms.appo.train_appo --env=doom_dwango5_bots_experimental --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.99 --env_frameskip=4 --rollout=32 --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048',
        _params.generate_params(randomize=False),
    ),

    Experiment(
        'bots_96_fs2',
        'python -m algorithms.appo.train_appo --env=doom_dwango5_bots_experimental --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --rollout=32 --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --res_w=96 --res_h=72 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),

    Experiment(
        'bots_96_fs4',
        'python -m algorithms.appo.train_appo --env=doom_dwango5_bots_experimental --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.99 --env_frameskip=4 --rollout=32 --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --res_w=96 --res_h=72 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription('doom_bots_v57_sweep', experiments=_experiments, pause_between_experiments=10, use_gpus=4, experiments_per_gpu=1, max_parallel=4)
