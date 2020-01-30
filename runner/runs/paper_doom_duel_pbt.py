from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ppo_epochs', [1]),
])

_experiment = Experiment(
    'bots_ssl2_fs2',
    'algorithms.appo.train_appo --env=doom_duel --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=80 --num_envs_per_worker=12 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --benchmark=False --pbt_replace_reward_gap=0.3 --pbt_replace_reward_gap_absolute=0.3 --pbt_period_env_steps=5000000 --save_milestones_sec=1800',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('paper_doom_duel_v65_fs2', experiments=[_experiment])
