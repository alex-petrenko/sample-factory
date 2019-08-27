from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['doom_basic']),
    ('recurrence', [1, 32]),
    ('use_rnn', [True, False]),
    ('new_clip', [True, False]),
])

_experiment = Experiment(
    'doom_sanity_v18',
    'python -m train_pytorch --train_for_seconds=240 --algo=PPO --ppo_clip_ratio=1.4 --ppo_epochs=8',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_sanity_v18', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)

run(gridsearch)
