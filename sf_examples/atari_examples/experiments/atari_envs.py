from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [0, 1111]),
        (
            "env",
            [
                "atari_breakout",
            ],
        ),
    ]
)

_experiments = [
    Experiment(
        "atari_test_run",
        "python -m sf_examples.atari_examples.train_atari --algo=APPO --with_wandb=True --wandb_user=wmFrank --wandb_project=atari-benchmark --wandb_tags test atari",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("atari_envs", experiments=_experiments)
# python -m sample_factory.runner.run --run=sf_examples.atari_examples.experiments.atari_envs --runner=processes --max_parallel=4  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
# python -m sample_factory.runner.run --run=sf_examples.atari_examples.experiments.atari_envs --runner=processes --max_parallel=4  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1 --experiment_suffix=4
