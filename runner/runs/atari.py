from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['atari_breakout', 'atari_gravitar', 'atari_spaceinvaders']),
    ('should_clip_hidden_states', ['True', 'False']),
    ('clip_hidden_states', [0.1]),
    ('hidden_states_clip_global', ['True']),
    ('ppo_epochs', [1, 4]),
])

_experiment = Experiment(
    'atari',
    'python -m train_pytorch --train_for_seconds=360000 --algo=PPO',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('atari_v33_clip', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)
