from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

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
        "doom_basic_envs",
        "python -m sf_examples.vizdoom_examples.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=16 --num_policies=1 --num_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --with_wandb=True --wandb_tags doom basic sf2",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("doom_basic_envs", experiments=_experiments)
# python -m sample_factory.runner.run --run=sf_examples.vizdoom_examples.experiments.doom_basic_envs --runner=processes --max_parallel=1  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1 --experiment_suffix=1
