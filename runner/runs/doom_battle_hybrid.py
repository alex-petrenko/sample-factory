from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('recurrence', [1, 16]),
    ('ppo_epochs', [1, 4, 8]),
])

_experiment = Experiment(
    'battle_v9_fs4',
    'python -m train_pytorch --env=doom_battle_hybrid --train_for_seconds=360000 --algo=PPO',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_battle_torch_v9_fs4_ent', experiments=[_experiment], pause_between_experiments=10, use_gpus=3, experiments_per_gpu=2, max_parallel=6)

run(gridsearch)
