from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('use_rnn', ['False']),
    ('recurrence', [1]),
    ('new_clip', ['True', 'False']),
    ('ppo_epochs', [1, 4, 8]),
])

_experiment = Experiment(
    'battle_v18_fs4_clip',
    'python -m train_pytorch --env=doom_battle_hybrid --train_for_seconds=360000 --algo=PPO --seed=42',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_battle_torch_v18_fs4_clip', experiments=[_experiment], pause_between_experiments=10, use_gpus=3, experiments_per_gpu=2, max_parallel=6)

run(gridsearch)
