from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['doom_battle_discrete', 'doom_battle_hybrid']),
    ('recurrence', [1, 16]),
])

_experiment = Experiment(
    'doom_tuple_actions',
    'python -m algorithms.ppo.train_ppo --train_for_seconds=360000 --gamma=0.995 --batch_size=1024 --entropy_loss_coeff=0.0005 --max_grad_norm=8.0 --rollout=128 --num_envs=144 --num_workers=18 --hidden_size=512 --ppo_clip_value=0.25',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_tuple_actions', experiments=[_experiment], pause_between_experiments=60)

run(gridsearch)
