from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [42]),
    ]
)

_experiments = [
    Experiment(
        "bots_128_fs2_narrow",
        "python -m sf_examples.vizdoom.train_vizdoom --env=doom_deathmatch_bots --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --num_workers=80 --num_envs_per_worker=24 --num_policies=8 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --with_pbt=True --pbt_period_env_steps=5000000",
        _params.generate_params(randomize=False),
        dict(DOOM_DEFAULT_UDP_PORT=35300),
    ),
]

RUN_DESCRIPTION = RunDescription("doom_bots_v100_pbt", experiments=_experiments)
