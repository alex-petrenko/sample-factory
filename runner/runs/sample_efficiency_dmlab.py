from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['dmlab_contributed/dmlab30/explore_goal_locations_small', 'dmlab_contributed/dmlab30/lasertag_three_opponents_small']),
    ('num_envs_per_worker', [10]),
])

_experiments = [
    Experiment(
        'dmlab_fs4',
        'algorithms.appo.train_appo --train_for_env_steps=200000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --policy_workers_per_policy=1 --experiment_summaries_interval=5 --hidden_size=256 --rnn_type=lstm --nonlinearity=relu',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('sample_efficiency_dmlab_v97_fs4_v2', experiments=_experiments)
