from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [000]),
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
        "atari_test_run",
        "python -m sf_examples.atari_examples.experiments.benchmark_atari --algo=APPO --with_wandb=True --wandb_user=wmFrank --wandb_project=atari-benchmark --wandb_group=atari_all --wandb_tags atari run2",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("atari_envs", experiments=_experiments)
# python -m sample_factory.runner.run --run=sf_examples.atari_examples.experiments.atari_envs --runner=processes --max_parallel=8  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
