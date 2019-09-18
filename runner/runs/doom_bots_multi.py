from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('ppo_epochs', [1]),
    ('early_stopping', ['False', 'True']),
    ('batch_size', [1024, 2048]),
])

_experiment = Experiment(
    'bots_multi_fs2',
    'python -m train_pytorch --env=doom_dwango5_multi --train_for_seconds=360000 --algo=PPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --rollout=64 --num_envs=96 --reward_scale=0.5',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_bots_multi_v40_fs2', experiments=[_experiment], pause_between_experiments=10, use_gpus=1, experiments_per_gpu=2, max_parallel=12)
