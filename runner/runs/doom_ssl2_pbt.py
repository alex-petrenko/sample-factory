from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ppo_epochs', [1]),
])

_experiment = Experiment(
    'bots_ssl2_fs2',
    'python -m algorithms.appo.train_appo --env=doom_ssl2_duel --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=80 --num_envs_per_worker=12 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --benchmark=False --pbt_replace_reward_gap=0.3 --pbt_period_env_steps=5000000',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_ssl2_duel_v64_fs2', experiments=[_experiment], pause_between_experiments=100, use_gpus=4, experiments_per_gpu=-1, max_parallel=1)
