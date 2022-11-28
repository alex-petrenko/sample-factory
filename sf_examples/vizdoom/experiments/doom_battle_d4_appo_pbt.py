from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("num_epochs", [1]),
    ]
)

_experiments = [
    Experiment(
        "battle_d4_fs4_pbt",
        "python -m sf_examples.vizdoom.train_vizdoom --env=doom_battle_d4 --train_for_env_steps=50000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=24 --num_envs_per_worker=30 --num_policies=4 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False --pbt_period_env_steps=5000000",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription(
    "doom_battle_d4_appo_v64_fs4_pbt", experiments=_experiments, use_gpus=2, experiments_per_gpu=-1, max_parallel=1
)
