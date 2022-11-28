def mujoco_override_defaults(env, parser):
    parser.set_defaults(
        batched_sampling=False,
        num_workers=8,
        num_envs_per_worker=8,
        worker_num_splits=2,
        train_for_env_steps=10000000,
        encoder_mlp_layers=[64, 64],
        env_frameskip=1,
        nonlinearity="tanh",
        batch_size=1024,
        kl_loss_coeff=0.1,
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization="torch_default",
        reward_scale=1,
        rollout=64,
        max_grad_norm=3.5,
        num_epochs=2,
        num_batches_per_epoch=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=1.3,
        exploration_loss_coeff=0.0,
        learning_rate=0.00295,
        lr_schedule="linear_decay",
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        recurrence=1,
        normalize_input=True,
        normalize_returns=True,
        value_bootstrap=True,
        experiment_summaries_interval=3,
        save_every_sec=15,
        serial_mode=False,
        async_rl=False,
    )


# noinspection PyUnusedLocal
def add_mujoco_env_args(env, parser):
    # in case we need to add more args in the future
    pass
