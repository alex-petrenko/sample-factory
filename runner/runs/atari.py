from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('env', ['atari_pong', 'atari_breakout', 'atari_gravitar', 'atari_spaceinvaders']),
])

_experiment = Experiment(
    'atari',
    'python -m train_pytorch --train_for_seconds=360000 --algo=PPO --recurrence=1 --use_rnn=False --ppo_epochs=10 --normalize_advantage=False --ppo_clip_value=10.0',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('atari_v30', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)
