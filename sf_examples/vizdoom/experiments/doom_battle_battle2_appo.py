from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription
from sample_factory.runner.runs.run_utils import seeds

_params = ParamGrid(
    [
        ("env", ["doom_battle", "doom_battle2"]),
        ("seed", seeds(4)),
    ]
)

_experiments = [
    Experiment(
        "battle_fs4",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("doom_battle_battle2_appo_v1.121.2", experiments=_experiments)
