from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('use_rnn', ['False']),
    ('recurrence', [1]),
    ('new_clip', ['True', 'False']),
    ('leaky_ppo', [0.5, 0.1, 0.0]),
    ('ppo_epochs', [4, 8]),
])

_experiment = Experiment(
    'battle_v19_fs4_leak',
    'python -m train_pytorch --env=doom_battle_hybrid --train_for_seconds=360000 --algo=PPO --seed=42',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_battle_torch_v19_fs4_leak', experiments=[_experiment], pause_between_experiments=10, use_gpus=6, experiments_per_gpu=2, max_parallel=12)

run(gridsearch)
