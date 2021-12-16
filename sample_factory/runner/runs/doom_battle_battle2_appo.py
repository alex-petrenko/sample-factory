from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from sample_factory.runner.runs.run_utils import seeds

_params = ParamGrid([
    ('seed', seeds(4)),
    ('env', ['doom_battle', 'doom_battle2']),
])

_experiments = [
    Experiment(
        'battle_fs4',
        'python -m sample_factory.algorithms.appo.train_appo --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('doom_battle_battle2_appo_v1.121.2', experiments=_experiments)
