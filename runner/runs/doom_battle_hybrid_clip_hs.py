from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('clip_hidden_states', [100.0, 0.1, 0.05, 0.025]),
])

_experiment = Experiment(
    'battle_fs4',
    'python -m train_pytorch --env=doom_battle_hybrid --train_for_seconds=360000 --algo=PPO --use_rnn=True --rollout=64 --num_envs=96 --reward_scale=0.5 --recurrence=32 --ppo_clip_ratio=1.1 --ppo_epochs=4',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_battle_torch_v29_fs4_clip', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)


