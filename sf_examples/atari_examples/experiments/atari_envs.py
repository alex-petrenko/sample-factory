from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [000, 111, 222, 333]),
        (
            "env",
            [
                "atari_breakout",
                # "atari_pong",
                # "atari_beamrider",
            ],
        ),
        # (
        #   "adam_eps", [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        # ),
    ]
)

_experiments = [
    Experiment(
        "atari_test_run",
        "python -m sf_examples.atari_examples.experiments.benchmark_atari --algo=APPO --with_wandb=True --wandb_user=wmFrank --wandb_project=atari-benchmark --wandb_group=atari_breakout --wandb_tags run5",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("atari_envs", experiments=_experiments)
# python -m sample_factory.runner.run --run=sf_examples.atari_examples.experiments.atari_envs --runner=processes --max_parallel=8  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
