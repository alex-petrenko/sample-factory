from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('env', ['doom_two_colors_easy', 'doom_two_colors_easy_no_input']),
    ('recurrence', [64]),
    ('use_rnn', [True, False]),
    ('ppo_epochs', [4]),
    ('mem_size', [4, 0]),
])

_experiment = Experiment(
    'mem_doom_v21',
    'python -m train_pytorch --algo=PPO --rollout=64 --num_envs=64 --train_for_env_steps=1000000000 --normalize_advantage=False --prior_loss_coeff=0.005',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('mem_doom_v21', experiments=[_experiment], pause_between_experiments=10, use_gpus=4, experiments_per_gpu=2, max_parallel=8)

run(gridsearch)
