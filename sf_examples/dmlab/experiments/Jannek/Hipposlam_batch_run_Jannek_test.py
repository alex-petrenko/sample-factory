from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription


_params = ParamGrid(
    [
        (
            ("Hippo_L","rnn_size"),
            (
                [ 64, 1149],
                
            )
        ),
        ("seed", [5555]),
    ]
)

# _params = ParamGrid(
#     [
#         # ("seed", [ 5555]),
#         # (
#         #     ("Hippo_L","rnn_size"),
#         #     (

#         #         [ 64, 1149]
                
#         #     )
#         # ),
#         # ("num_envs_per_worker",[2,4,8]),
#         # ("batch_size",[1024,2048]),
#         # ("num_batches_per_epoch",[2,4]),

#         # ("env_frameskip",[8,4]),
#         ("seed", [0, 1111, 2222, 7777, 8888, 9999]),

#         # ("num_policies",[4,8,16]),
#         (
#             ("Hippo_L","rnn_size"),
#             (
#                 [ 1 , 141],
#                 [ 8 , 253],
#                 [ 16, 381],
#                 [ 32, 637],
#                 [ 64, 1149]
                
#             )
#         ),
#     ]
# )


vstr = "hipposlam"

cli = (
    "--env=openfield_map2_fixed_loc3 "
    "--wandb_project=SF_dmlab_jannek_batchRUN "
    "--seed=42 "
    "--train_for_seconds=300 "
    "--algo=APPO "
    "--gamma=0.99 "
    "--use_rnn=True "
    "--num_workers=32 "
    "--num_envs_per_worker=8 "
    "--worker_num_splits=8 "
    "--num_epochs=1 "
    "--rollout=64 "
    "--recurrence=64 "
    "--batch_size=2048 "
    "--num_batches_per_epoch=2 "
    "--benchmark=False "
    "--max_grad_norm=0.0 "
    "--dmlab_renderer=software "
    "--decorrelate_experience_max_seconds=120 "
    "--nonlinearity=relu "
    "--rnn_type=gru "
    "--dmlab_extended_action_set=False "
    "--encoder_conv_architecture=pretrained_resnet "
    "--encoder_conv_mlp_layers=256 "
    "--dmlab_one_task_per_worker=True "
    "--set_workers_cpu_affinity=False "
    "--dmlab_use_level_cache=True "
    "--num_policies=8 "
    "--pbt_replace_reward_gap=0.01 "
    "--pbt_replace_reward_gap_absolute=0.15 "
    "--pbt_period_env_steps=2000000 "
    "--pbt_start_mutation=10000000 "
    "--with_pbt=True "
    "--max_policy_lag=35 "
    "--use_record_episode_statistics=True "
    "--keep_checkpoints=10 "
    "--save_every_sec=120 "
    "--save_milestones_sec=5400 "
    "--decoder_mlp_layers 128 128 "
    "--env_frameskip=4 "
    "--dmlab_reduced_action_set=True "
    "--core_name=BypassSS "
    "--rnn_type=gru "
    "--DG_name=batchnorm_relu "
    "--learning_rate=0.0002 "
    "--fix_encoder_when_load=True "
    "--encoder_load_path=/home/fr/fr_js1764/trainingJ/best_000025288_203030528_reward_94.185.pth "
    "--with_wandb=True "
    "--wandb_user=xiaoxionglin-bernstein-center-freiburg "
    "--pbt_mix_policies_in_one_env=False "
    "--pbt_target_objective=lenweighted_score "
    "--with_number_instruction=True "
    "--save_best_metric=lenweighted_score "
    "--device=cpu "
    "--Hippo_n_feature=16 "
    "--number_instruction_coef=9 "
    "--DG_BN_intercept=2.43 "
    "--depth_sensor=True "
    "--normalize_input=False "
    "--Hippo_L=64 "
    "--rnn_size=1149 "
    "--exploration_loss_coeff=0.005 "
    "--value_loss_coeff=0.3 "
    "--ppo_clip_ratio=0.25 "
    "--pbt_perturb_max=1.3 "
    "--pbt_replace_fraction=0.2 "
    "--save_best_every_sec=30 "
)


_experiments = [
    Experiment("Grid_hippoL", cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f"{vstr}", experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --backend=processes --max_parallel=1 --experiments_per_gpu=1 --num_gpus=1 --run=sf_examples.dmlab.experiments.dmlab30
# Run on Slurm: python -m sample_factory.launcher.run --backend=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False --run=sf_examples.dmlab.experiments.dmlab30
# python -m sample_factory.launcher.run --backend=slurm --slurm_workdir=./slurm_grid --slurm_gpus_per_job=0 --slurm_cpus_per_gpu=50 --slurm_sbatch_template=./training_template.sh --pause_between=1 --slurm_print_only=False --run=sf_examples.dmlab.experiments.Hipposlam_batch_run_Hippo_L --slurm_partition=genoa --slurm_timeout=40:10:00