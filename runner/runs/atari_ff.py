from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['atari_breakout', 'atari_spaceinvaders', 'atari_qbert']),
    ('normalize_advantage', ['False', 'True']),
])

_experiment = Experiment(
    'atari_ff',
    'python -m train_pytorch --train_for_seconds=360000 --algo=PPO --train_for_env_steps=200000000 --recurrence=1 --use_rnn=False --ppo_epochs=10',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('atari_v33_ff', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)
