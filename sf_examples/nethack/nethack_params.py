from sample_factory.utils.utils import str2bool


def add_extra_params_nethack_env(parser):
    """
    Specify any additional command line arguments for NetHack environments.
    """
    # TODO: add help
    p = parser
    p.add_argument("--character", type=str, default="@")
    p.add_argument("--max_episode_steps", type=int, default=100000)
    p.add_argument("--penalty_step", type=float, default=0.0)
    p.add_argument("--penalty_time", type=float, default=0.0)
    p.add_argument("--fn_penalty_step", type=str, default="constant")
    p.add_argument("--savedir", type=str, default=None)
    p.add_argument("--save_ttyrec_every", type=int, default=0)
    p.add_argument("--state_counter", type=str, default=None)
    p.add_argument("--add_image_observation", type=str2bool, default=True)
    p.add_argument("--crop_dim", type=int, default=18)
    p.add_argument("--pixel_size", type=int, default=6)


def add_extra_params_model(parser):
    """
    Specify any additional command line arguments for NetHack models.
    """
    # TODO: add help
    p = parser
    p.add_argument("--use_prev_action", type=str2bool, default=False)
    p.add_argument("--h_dim", type=int, default=512)
    p.add_argument("--msg_hdim", type=int, default=64)
    p.add_argument("--color_edim", type=int, default=16)
    p.add_argument("--char_edim", type=int, default=16)
    p.add_argument("--use_crop", type=str2bool, default=True)
    p.add_argument("--use_crop_norm", type=str2bool, default=True)
    p.add_argument("--screen_kernel_size", type=int, default=3)
    p.add_argument("--no_max_pool", type=str2bool, default=False)
    p.add_argument("--screen_conv_blocks", type=int, default=2)
    p.add_argument("--blstats_hdim", type=int, default=512)
    p.add_argument("--fc_after_cnn_hdim", type=int, default=512)
    p.add_argument("--use_resnet", type=str2bool, default=False)


def add_extra_params_general(parser):
    """
    Specify any additional command line arguments for NetHack.
    """
    # TODO: add help
    p = parser
    p.add_argument("--model", type=str, default="ChaoticDwarvenGPT5")
    p.add_argument("--add_stats_to_info", type=str2bool, default=True)


def nethack_override_defaults(_env, parser):
    """RL params specific to NetHack envs."""
    # TODO:
    parser.set_defaults(
        use_record_episode_statistics=False,
        gamma=0.999,  # discounting
        num_workers=12,
        num_envs_per_worker=2,
        worker_num_splits=2,
        train_for_env_steps=2_000_000_000,
        nonlinearity="relu",
        use_rnn=True,
        rnn_type="lstm",
        actor_critic_share_weights=True,
        policy_initialization="orthogonal",
        policy_init_gain=1.0,
        adaptive_stddev=False,  # True only for continous action distributions
        reward_scale=1.0,
        reward_clip=10.0,  # tune?? 30?
        batch_size=1024,
        rollout=32,
        max_grad_norm=4,  # TODO: search
        num_epochs=1,  # TODO: in some atari - 4, maybe worth checking
        num_batches_per_epoch=1,  # can be used for increasing the batch_size for SGD
        ppo_clip_ratio=0.1,  # TODO: tune
        ppo_clip_value=1.0,
        value_loss_coeff=1.0,  # TODO: tune
        exploration_loss="entropy",
        exploration_loss_coeff=0.001,  # TODO: tune
        learning_rate=0.0001,  # TODO: tune
        # lr_schedule="linear_decay", # TODO: test later
        gae_lambda=1.0,  # TODO: here default 0.95, 1.0 means we turn off gae
        with_vtrace=False,
        normalize_input=False,  # TODO: turn off for now and use normalization from d&d, then switch and check what happens
        normalize_returns=True,  # TODO: we should check what works better, normalized returns or vtrace
        async_rl=True,
        experiment_summaries_interval=50,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-7,
        seed=22,
        save_every_sec=120,
    )
