from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [42]),
    ]
)

_experiments = [
    Experiment(
        "bots_128_fs2_wide",
        "python -m sf_examples.vizdoom.train_vizdoom --env=doom_dwango5_bots_experimental --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --reward_scale=0.5 --num_workers=18 --num_envs_per_worker=20 --num_policies=1 --macro_batch=2048 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=True",
        _params.generate_params(randomize=False),
        dict(DOOM_DEFAULT_UDP_PORT=35300),
    ),
    Experiment(
        "bots_128_fs2_narrow",
        "python -m sf_examples.vizdoom.train_vizdoom --env=doom_dwango5_bots_experimental --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --reward_scale=0.5 --num_workers=18 --num_envs_per_worker=20 --num_policies=1 --macro_batch=2048 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False",
        _params.generate_params(randomize=False),
        dict(DOOM_DEFAULT_UDP_PORT=40300),
    ),
    Experiment(
        "bots_128_fs2_wide_adam0.5",
        "python -m sf_examples.vizdoom.train_vizdoom --env=doom_dwango5_bots_experimental --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --reward_scale=0.5 --num_workers=18 --num_envs_per_worker=20 --num_policies=1 --macro_batch=2048 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=True --adam_beta1=0.5",
        _params.generate_params(randomize=False),
        dict(DOOM_DEFAULT_UDP_PORT=45300),
    ),
    Experiment(
        "bots_128_fs2_narrow_adam0.5",
        "python -m sf_examples.vizdoom.train_vizdoom --env=doom_dwango5_bots_experimental --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --reward_scale=0.5 --num_workers=18 --num_envs_per_worker=20 --num_policies=1 --macro_batch=2048 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --adam_beta1=0.5",
        _params.generate_params(randomize=False),
        dict(DOOM_DEFAULT_UDP_PORT=50300),
    ),
]

RUN_DESCRIPTION = RunDescription(
    "doom_bots_sweep",
    experiments=_experiments,
)
