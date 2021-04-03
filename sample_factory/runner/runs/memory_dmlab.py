from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('use_rnn', [True, False]),
    ('mem_size', [4, 0]),
])

cmd = 'python -m train_pytorch --algo=PPO --rollout=64 --recurrence=32 --num_envs=96 --num_workers=96 --train_for_env_steps=1000000000 --normalize_advantage=False --prior_loss_coeff=0.005 '

# IMPORTANT: for DMLAB number of workers better be equal to the number of envs, otherwise spurious crashes may occur!
_experiment_nm = Experiment(
    'mem_dmlab_nm',
    cmd + '--reward_scale=0.1 --env=dmlab_nonmatch',
    _params.generate_params(randomize=False),
)
_experiment_wm = Experiment(
    'mem_dmlab_wm',
    cmd + '--reward_scale=1.0 --env=dmlab_watermaze',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('mem_dmlab_v24', experiments=[_experiment_nm, _experiment_wm])
