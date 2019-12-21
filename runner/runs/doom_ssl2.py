from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ppo_epochs', [1]),
    ('batch_size', [1024]),
    ('learning_rate', [1e-4, 2e-4, 4e-4]),
])

_experiment = Experiment(
    'bots_ssl2_fs2',
    'python -m algorithms.appo.train_appo --env=doom_ssl2_duel --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --rollout=64 --num_envs=96 --reward_scale=0.5 --num_workers=60 --num_envs_per_worker=10 --num_policies=4 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=1024 --experiment=doom_ssl2_4p_v49 --benchmark=False --max_grad_norm=0.0',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_ssl2_duel_v42_fs2', experiments=[_experiment], pause_between_experiments=10, use_gpus=2, experiments_per_gpu=2, max_parallel=4)
