from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_experiment = Experiment(
    "bots_freedm_fs2",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_freedm --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=4 --num_policies=1 --macro_batch=2048 --batch_size=2048 --benchmark=False --start_bot_difficulty=150",
)

RUN_DESCRIPTION = RunDescription(
    "doom_freedm_fs2",
    experiments=[_experiment],
)
