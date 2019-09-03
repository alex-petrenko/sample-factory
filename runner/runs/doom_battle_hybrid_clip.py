from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('ppo_clip_ratio', [1.1]),
    ('ppo_epochs', [4]),
    ('rnn_dist_loss_coeff', [0.1, 0.01, 0.0, 1.0]),
])

_experiment = Experiment(
    'battle_fs4',
    'python -m train_pytorch --env=doom_battle_hybrid --train_for_seconds=360000 --algo=PPO --use_rnn=True --rollout=64 --num_envs=96 --reward_scale=0.5 --recurrence=32',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_battle_torch_v28_fs4_rnndist', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)


