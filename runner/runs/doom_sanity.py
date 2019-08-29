from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('env', ['doom_basic']),
    ('recurrence', [1]),
    ('use_rnn', [False]),
    ('new_clip', [True, False]),
    ('leaky_ppo', [0.0, 0.5]),
])

_experiment = Experiment(
    'doom_sanity_v19',
    'python -m train_pytorch --train_for_seconds=240 --algo=PPO --ppo_clip_ratio=1.4 --ppo_epochs=8 --seed=42',
    _params.generate_params(randomize=False),
)

run_description = RunDescription('doom_sanity_v19', experiments=[_experiment], pause_between_experiments=5, use_gpus=2, experiments_per_gpu=2, max_parallel=4)

run(run_description)
