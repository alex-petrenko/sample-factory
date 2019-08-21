from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['doom_corridor']),
    ('entropy_loss_coeff', [0.0005, 0.005, 0.001]),
    ('normalize_advantage', ['False', 'True']),
])

_experiment = Experiment(
    'doom_sanity_v10',
    'python -m train_pytorch --train_for_seconds=3600 --algo=PPO',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_sanity_v10', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)

run(gridsearch)
