from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('seed', [42, 42, 42, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]),
])

_experiment = Experiment(
    'battle_v20_seeds',
    'python -m train_pytorch --env=doom_battle_hybrid --train_for_seconds=360000 --algo=PPO --use_rnn=False --recurrence=1',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_battle_torch_v20', experiments=[_experiment], pause_between_experiments=10, use_gpus=6, experiments_per_gpu=2, max_parallel=12)

run(gridsearch)
