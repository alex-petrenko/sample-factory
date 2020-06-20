from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['doom_two_colors_easy', 'doom_two_colors_hard']),
    ('recurrence', [1, 16]),
    ('batch_size', [512, 1024]),
])

_experiment = Experiment(
    'doom_two_colors_grid',
    'python -m algorithms.ppo.train_ppo --train_for_seconds=36000',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_two_colors_grid', experiments=[_experiment])
