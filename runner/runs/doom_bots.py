from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('ppo_clip_ratio', [1.1, 1.3]),
    ('ppo_epochs', [1, 2, 4]),
    ('recurrence', [16, 32]),
])

_experiment = Experiment(
    'bots_fs2',
    'python -m train_pytorch --env=doom_dwango5_bots_experimental --train_for_seconds=360000 --algo=PPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --rollout=64 --num_envs=96 --reward_scale=0.5',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_bots_v27_fs2_seeds', experiments=[_experiment], pause_between_experiments=10, use_gpus=6, experiments_per_gpu=2, max_parallel=12)
