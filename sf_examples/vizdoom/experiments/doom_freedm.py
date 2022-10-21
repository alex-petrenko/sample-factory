from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("num_epochs", [1]),
    ]
)

_experiment = Experiment(
    "bots_freedm_fs2",
    "python -m sf_examples.vizdoom_examples.train_vizdoom --env=doom_freedm --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=4 --num_policies=1 --macro_batch=2048 --batch_size=2048 --benchmark=False --start_bot_difficulty=150",
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription(
    "doom_freedm_v64_fs2",
    experiments=[_experiment],
    pause_between_experiments=100,
    use_gpus=1,
    experiments_per_gpu=-1,
    max_parallel=1,
)
