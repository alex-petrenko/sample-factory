from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid([])

_experiments = [
    Experiment(
        "battle_fs4",
        "python -m sf_examples.vizdoom.train_vizdoom --env=doom_battle --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --batch_size=2048 --wide_aspect_ratio=False --num_workers=72 --num_envs_per_worker=32 --num_policies=8 --with_pbt=True",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("paper_doom_battle_appo_pbt_v98_fs4", experiments=_experiments)
