from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [00, 11, 22, 33]),
        (
            "env",
            [
                "atari_breakout",
                "atari_pong",
                "atari_beamrider",
            ],
        ),
    ]
)

_experiments = [
    Experiment(
        "atari_envs",
        "python -m sf_examples.atari.train_atari --algo=APPO --with_wandb=True --wandb_project=atari-benchmark --wandb_group=atari_all --wandb_tags run6",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("atari_envs", experiments=_experiments)
# python -m sample_factory.launcher.run --run=sf_examples.atari.experiments.atari_envs --backend=processes --max_parallel=8  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
