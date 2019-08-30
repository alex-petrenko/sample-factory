from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('ppo_clip_ratio', [1.1, 1.3]),
    ('ppo_epochs', [1, 2, 4]),
    ('recurrence', [16, 32]),
])

_experiment = Experiment(
    'bots_v20_fs2',
    'python -m train_pytorch --env=doom_dwango5_bots_experimental --train_for_seconds=360000 --algo=PPO --gamma=0.995 --recurrence=1 --use_rnn=False --env_frameskip=2',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_bots_v20_fs2_seeds', experiments=[_experiment], pause_between_experiments=10, use_gpus=6, experiments_per_gpu=2, max_parallel=12)

run(gridsearch)
