from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription


_params = ParamGrid(
    [
<<<<<<< HEAD
        ("seed", [0,1111]),#, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
        (
            ("Hippo_L","rnn_size"),
            [( 1 , 141),
             ( 8 , 253)
=======
        ("seed", [0]),#, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
        (
            ("Hippo_L","rnn_size"),
            ([ 1 , 141],
            #  [ 8 , 253],
>>>>>>> hipposlam_sf_github_https/main
            #  [ 16, 381],
            #  [ 32, 637],
            #  [ 64, 1149]
                
<<<<<<< HEAD
            ]
=======
            )
>>>>>>> hipposlam_sf_github_https/main
        ),
    ]
)


vstr = "hipposlam_search"

cli = (
    "--env=openfield_map2_fixed_loc3 "
    "--train_for_seconds=72000 "
    "--algo=APPO "
    "--gamma=0.99 "
    "--use_rnn=True "
    "--num_workers=32 "
<<<<<<< HEAD
    "--num_envs_per_worker=2 "
=======
    "--num_envs_per_worker=4 "
>>>>>>> hipposlam_sf_github_https/main
    "--num_epochs=1 "
    "--rollout=64 "
    "--recurrence=64 "
    "--batch_size=1536 "
    "--benchmark=False "
    "--max_grad_norm=0.0 "
    "--dmlab_renderer=software "
    "--decorrelate_experience_max_seconds=90 "
    "--nonlinearity=relu "
    "--rnn_type=gru "
    "--dmlab_extended_action_set=False "
<<<<<<< HEAD
    "--encoder_conv_architecture=pretrained_resnet "
    "--encoder_conv_mlp_layers=256 "
    "--dmlab_one_task_per_worker=True "
    "--set_workers_cpu_affinity=True "
    "--dmlab_use_level_cache=True "
    "--num_policies=4 "
    "--pbt_replace_reward_gap=0.05 "
    "--pbt_replace_reward_gap_absolute=0.2 "
    "--pbt_period_env_steps=1000000 "
    "--pbt_start_mutation=5000000 "
    "--with_pbt=True "
    "--max_policy_lag=35 "
    "--use_record_episode_statistics=True "
    "--keep_checkpoints=10 "
    "--save_every_sec=120 "
    "--save_milestones_sec=4000 "
    "--decoder_mlp_layers 128 128 "
    "--env_frameskip=8 "
    "--dmlab_reduced_action_set=True "
    "--core_name=BypassSS "
    "--rnn_type=gru "
    "--DG_name=batchnorm_relu "
    "--learning_rate=0.0001 "
    "--fix_encoder_when_load=True "
    "--encoder_load_path=/home/fr/fr_xl1014/training/best_000025288_203030528_reward_94.185.pth "
    "--with_wandb=True "
    "--wandb_user=xiaoxionglin-bernstein-center-freiburg "
    "--pbt_mix_policies_in_one_env=False "
    "--wandb_project=SF_dmlab_NEMO_grid "
    "--worker_num_splits=2 "
    "--pbt_target_objective=lenweighted_score "
    "--with_number_instruction=True "
    "--save_best_metric=avg_z_00_openfield_map2_fixed_loc3_lenweighted_score "
    "--device=cpu "
    "--Hippo_n_feature=16 "
    "--number_instruction_coef=9 "
    "--DG_BN_intercept=2.43 "
    "--depth_sensor=True "
    "--exploration_loss_coeff=0.008 "
    "--value_loss_coeff=0.15 "
    "--ppo_clip_ratio=0.35"
)

=======
    "--encoder_conv_architecture=pretrained_resnet",
    "--encoder_conv_mlp_layers=256",
    "--dmlab_one_task_per_worker=True",
    "--set_workers_cpu_affinity=True",
    "--dmlab_use_level_cache=True",
    "--num_policies=4",
    "--pbt_replace_reward_gap=0.05",
    "--pbt_replace_reward_gap_absolute=0.2",
    "--pbt_period_env_steps=1000000",
    "--pbt_start_mutation=5000000",
    "--with_pbt=True",
    "--max_policy_lag=35",
    "--use_record_episode_statistics=True",
    "--keep_checkpoints=10",
    "--save_every_sec=120",
    "--save_milestones_sec=4000",
    "--decoder_mlp_layers", "128", "128",
    # "--Hippo_L=64",
    "--env_frameskip=8",
    "--dmlab_reduced_action_set=True",
    "--core_name=BypassSS",
    # f"--rnn_size={16*(1149",
    "--rnn_type=gru",
    "--DG_name=batchnorm_relu",
    "--learning_rate=0.0001",
    "--fix_encoder_when_load=True",
    "--encoder_load_path=/home/fr/fr_xl1014/training/best_000025288_203030528_reward_94.185.pth",
    "--with_wandb=True",
    "--wandb_user=xiaoxionglin-bernstein-center-freiburg",
    "--pbt_mix_policies_in_one_env=False",
    "--wandb_project=SF_dmlab_NEMO_grid",
    "--worker_num_splits=2",
    "--pbt_target_objective=lenweighted_score",
    "--with_number_instruction=True",
    "--save_best_metric=avg_z_00_openfield_map2_fixed_loc3_lenweighted_score",
    "--device=cpu",
    "--Hippo_n_feature=16",
    "--number_instruction_coef=9",
    "--DG_BN_intercept=2.43",
    "--depth_sensor=True",
    # "--seed=42",
    "--exploration_loss_coeff=0.008",
    "--value_loss_coeff=0.15",
    "--ppo_clip_ratio=0.35"
)
>>>>>>> hipposlam_sf_github_https/main
_experiments = [
    Experiment("hipposlam_", cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f"{vstr}", experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --backend=processes --max_parallel=1 --experiments_per_gpu=1 --num_gpus=1 --run=sf_examples.dmlab.experiments.dmlab30
# Run on Slurm: python -m sample_factory.launcher.run --backend=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False --run=sf_examples.dmlab.experiments.dmlab30
