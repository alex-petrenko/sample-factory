from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [1111, 2222, 3333, 4444, 5555]),
        ("env", ["doom_health_gathering_supreme"]),
    ]
)

_experiments = [
    Experiment(
        "health_0_255",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=40000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=12 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False",
        _params.generate_params(randomize=False),
    ),
    Experiment(
        "health_128_128",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=40000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=12 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --obs_subtract_mean=128.0 --obs_scale=128.0",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("doom_health_gathering_v97_fs4", experiments=_experiments)
