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
        "doom_battle_envs",
        "python -m sf_examples.vizdoom_examples.train_vizdoom --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --num_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --with_wandb=True --wandb_tags doom battle sf2",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("doom_battle_envs", experiments=_experiments)
