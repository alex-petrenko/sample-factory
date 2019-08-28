from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['dmlab_watermaze']),
    ('recurrence', [64]),
    ('use_rnn', [True, False]),
    ('ppo_epochs', [4]),
    ('mem_size', [4, 0]),
])

# IMPORTANT: for DMLAB number of workers better be equal to the number of envs, otherwise spurious crashes may occur!
_experiment = Experiment(
    'mem_dmlab_v21',
    'python -m train_pytorch --algo=PPO --rollout=64 --num_envs=64 --num_workers=64 --train_for_env_steps=1000000000 --normalize_advantage=False --prior_loss_coeff=0.005',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('mem_dmlab_v21_wm', experiments=[_experiment], pause_between_experiments=15, use_gpus=4, experiments_per_gpu=1, max_parallel=4)

run(gridsearch)
