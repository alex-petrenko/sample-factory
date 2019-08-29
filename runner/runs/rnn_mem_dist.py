from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('rnn_dist_loss_coeff', [0.000, 0.01, 0.1]),
])

_experiment = Experiment(
    'mem_dist',
    'python -m algorithms.ppo.train_ppo --train_for_seconds=72000 --env=doom_two_colors_easy --recurrence=16',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('rnn_mem_dist', experiments=[_experiment], pause_between_experiments=30)

run(gridsearch)
