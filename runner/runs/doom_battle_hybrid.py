from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('recurrence', [1, 16]),
    ('ppo_epochs', [1, 4, 8]),
    ('ppo_clip_value', [0.1]),
])

_experiment = Experiment(
    'battle_v8_fs4',
    'python -m train_pytorch --env=doom_battle_hybrid --train_for_seconds=360000 --algo=PPO --env=doom_battle_hybrid --rollout=32 --num_envs=128 --num_workers=16 --ppo_clip_ratio=1.1 --batch_size=1024 --max_grad_norm=40.0 --gamma=0.99',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_battle_torch_v8_fs4_vc0.1', experiments=[_experiment], pause_between_experiments=10, use_gpus=3, experiments_per_gpu=2, max_parallel=6)

run(gridsearch)
