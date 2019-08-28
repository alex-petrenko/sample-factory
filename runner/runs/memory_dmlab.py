from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['dmlab_nonmatch', 'dmlab_watermaze']),
    ('recurrence', [64]),
    ('use_rnn', [True, False]),
    ('ppo_epochs', [2, 4]),
])

_experiment = Experiment(
    'mem_dmlab_v20',
    'python -m train_pytorch --algo=PPO --rollout=64 --num_envs=64 --train_for_env_steps=1000000000',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('mem_dmlab_v20', experiments=[_experiment], pause_between_experiments=10, use_gpus=4, experiments_per_gpu=2, max_parallel=8)

run(gridsearch)
