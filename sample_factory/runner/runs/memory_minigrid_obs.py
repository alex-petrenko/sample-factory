from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['MiniGrid-MemoryS7-v0', 'MiniGrid-RedBlueDoors-8x8-v0', 'MiniGrid-MemoryS17Random-v0']),
])

_experiment = Experiment(
    'mem_minigrid_obs',
    'python -m train_pytorch --algo=PPO --rollout=64 --num_envs=96 --recurrence=1 --use_rnn=False --train_for_env_steps=200000000 --prior_loss_coeff=0.005 --obs_mem=True',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('mem_minigrid_obs_v26', experiments=[_experiment], pause_between_experiments=5, use_gpus=2, experiments_per_gpu=4, max_parallel=12)
