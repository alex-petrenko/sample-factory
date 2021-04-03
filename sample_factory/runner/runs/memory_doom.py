from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['doom_two_colors_easy', 'doom_two_colors_hard']),
    ('use_rnn', [True, False]),
    ('mem_size', [4, 0]),
])

_experiment = Experiment(
    'mem_doom',
    'python -m train_pytorch --algo=PPO --train_for_env_steps=1000000000 --prior_loss_coeff=0.005 --reward_scale=0.5',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('mem_doom_v39', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)
