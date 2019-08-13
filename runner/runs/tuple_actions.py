from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('recurrence', [1, 16]),
    ('ppo_epochs', [1, 4]),
])

_experiment = Experiment(
    'doom_tuple_actions',
    'python -m algorithms.ppo.train_ppo --env=doom_battle_hybrid --train_for_seconds=360000 --gamma=0.998 --batch_size=1024 --entropy_loss_coeff=0.0005 --num_envs=144 --num_workers=18 --hidden_size=512 --env_frameskip=2',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_tuple_actions_v5', experiments=[_experiment], pause_between_experiments=30)

run(gridsearch)
