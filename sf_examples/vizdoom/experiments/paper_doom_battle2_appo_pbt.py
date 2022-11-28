from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid([])

_experiments = [
    Experiment(
        "battle2_fs4",
        "python -m sf_examples.vizdoom.train_vizdoom --env=doom_battle2 --train_for_env_steps=3000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False --num_workers=72 --num_envs_per_worker=30 --num_policies=8 --with_pbt=True",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("paper_doom_battle2_appo_pbt_v65_fs4", experiments=_experiments)
