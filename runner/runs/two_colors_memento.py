from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('use_rnn', [True, False]),
    ('recurrence', [32]),
    ('mem_size', [0, 4]),
    ('mem_feature', [32]),
])

_experiment = Experiment(
    'doom_two_colors_mem_v14',
    'python -m train_pytorch --algo=PPO --env=doom_two_colors_easy --rollout=64 --num_envs=64 --prior_loss_coeff=0.005',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_two_colors_mem_v14', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)

run(gridsearch)
