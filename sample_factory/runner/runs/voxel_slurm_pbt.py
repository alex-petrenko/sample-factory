from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ppo_epochs', [1]),
])

_experiment = Experiment(
    'voxel_env_pbt',
    'python -m algorithms.appo.train_appo --env=voxel_env_v23_v --train_for_seconds=360000000 --algo=APPO --gamma=0.997 --use_rnn=True --num_workers=28 --num_envs_per_worker=2 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --actor_worker_gpus 0 1 2 3 4 5 6 7 --num_policies=8 --with_pbt=True --max_grad_norm=0.0 --pbt_replace_reward_gap_absolute=0.3 --exploration_loss=symmetric_kl --exploration_loss_coeff=0.001 --pbt_mix_policies_in_one_env=False --experiment=voxel_env_v23_v --voxel_num_envs_per_instance=48 --voxel_num_agents_per_env=4 --voxel_num_simulation_threads=4 --voxel_vertical_look_limit=0.2 --voxel_use_vulkan=True --policy_workers_per_policy=2 --learner_main_loop_num_cores=4',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('voxel_env_pbt_v112_env_v23_8p', experiments=[_experiment])
