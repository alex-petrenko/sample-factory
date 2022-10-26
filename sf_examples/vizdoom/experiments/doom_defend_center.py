from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [1111, 2222, 3333]),
        ("env", ["doom_defend_the_center_flat_actions"]),
        ("num_envs_per_worker", [16]),
    ]
)

_experiments = [
    Experiment(
        "basic_envs_fs4",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=100000000 --algo=APPO --env_frameskip=4 --use_rnn=True --rnn_type=lstm --num_workers=72 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --policy_workers_per_policy=3 --experiment_summaries_interval=5 --ppo_clip_value=10.0 --nonlinearity=relu",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("paper_doom_wall_time_v97_fs4", experiments=_experiments)
