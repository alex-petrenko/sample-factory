from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid([])

_experiment = Experiment(
    "basic",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=3000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=20 --experiment=doom_basic",
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("doom_basic", experiments=[_experiment])
