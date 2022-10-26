from sample_factory.launcher.launcher_utils import seeds
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", seeds(4)),
    ]
)

_experiment = Experiment(
    "doom_basic",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=2000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False",
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("doom_basic", experiments=[_experiment])

# To run:
# python -m sample_factory.launcher.run --run=sf_examples.vizdoom.experiments.doom_basic --backend=processes --num_gpus=1 --max_parallel=2  --pause_between=0 --experiments_per_gpu=2
