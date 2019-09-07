from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['atari_breakout', 'atari_spaceinvaders', 'atari_qbert', 'atari_mspacman']),
    ('ppo_epochs', [10]),
    ('kl_coeff_large', [50.0, 0.0]),
])

_experiment = Experiment(
    'atari',
    'python -m train_pytorch --train_for_seconds=360000 --algo=PPO --train_for_env_steps=200000000',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('atari_v34_kl', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)
