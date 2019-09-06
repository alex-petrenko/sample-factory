from runner.run_description import RunDescription, Experiment, ParamGrid

_params_glob = ParamGrid([
    ('env', ['atari_breakout', 'atari_spaceinvaders', 'atari_qbert']),
    ('should_clip_hidden_states', ['True', 'False']),
    ('clip_hidden_states', [0.25, 0.5]),
    ('hidden_states_clip_global', ['True']),
    ('ppo_epochs', [10]),
])

_experiment_glob = Experiment(
    'atari_clip_glob',
    'python -m train_pytorch --train_for_seconds=360000 --algo=PPO --train_for_env_steps=200000000',
    _params_glob.generate_params(randomize=False),
)

# _params_elem = ParamGrid([
#     ('env', ['atari_breakout', 'atari_spaceinvaders']),
#     ('should_clip_hidden_states', ['True', 'False']),
#     ('clip_hidden_states', [0.025]),
#     ('hidden_states_clip_global', ['False']),
#     ('ppo_epochs', [1, 4]),
# ])
#
# _experiment_elem = Experiment(
#     'atari_clip_el',
#     'python -m train_pytorch --train_for_seconds=360000 --algo=PPO --train_for_env_steps=200000000',
#     _params_elem.generate_params(randomize=False),
# )

experiments = [_experiment_glob]

RUN_DESCRIPTION = RunDescription('atari_v33_clip', experiments=experiments, pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)
