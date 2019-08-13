from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['doom_battle_hybrid']),
    ('rollout', [64, 32]),
    ('recurrence', [1, 16]),
    ('ppo_clip_value', [0.25, 10.0]),
])

_experiment = Experiment(
    'doom_tuple_actions',
    'python -m algorithms.ppo.train_ppo --train_for_seconds=360000 --gamma=0.998 --batch_size=1024 --entropy_loss_coeff=0.0005 --max_grad_norm=8.0 --num_envs=144 --num_workers=18 --hidden_size=512 --env_frameskip=2',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_tuple_actions_v5', experiments=[_experiment], pause_between_experiments=30)

run(gridsearch)
