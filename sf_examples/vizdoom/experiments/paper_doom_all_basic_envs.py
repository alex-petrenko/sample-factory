from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
        (
            "env",
            [
                "doom_my_way_home",
                "doom_deadly_corridor",
                "doom_defend_the_center",
                "doom_defend_the_line",
                "doom_health_gathering",
                "doom_health_gathering_supreme",
            ],
        ),
    ]
)

_experiments = [
    Experiment(
        "basic_envs_fs4",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("paper_doom_basic_envs", experiments=_experiments)
