def mujoco_override_defaults(env, parser):
    parser.set_defaults(
        batched_sampling=False,
        num_workers=8,
        num_envs_per_worker=16,
        worker_num_splits=2,
        train_for_env_steps=1000000,

        encoder_type='mlp',
        encoder_subtype='mlp_mujoco',
        hidden_size=64,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
        nonlinearity='tanh',
        batch_size=1024,

        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization='torch_default',
        reward_scale=0.1,
        rollout=16,
        max_grad_norm=0.0,
        ppo_epochs=10,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.0,
        learning_rate=3e-4,
        lr_schedule='kl_adaptive_minibatch',
        lr_schedule_kl_threshold=0.08,
        shuffle_minibatches=True,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        recurrence=1,
        value_bootstrap=False,
        normalize_input=True,
        experiment_summaries_interval=3,
        save_every_sec=15,

        serial_mode=False,
    )

    # environment specific overrides
    env_name = '_'.join(env.split('_')[1:]).lower()

    # TODO: Hopper, Reacher, Swimmer, Walker
    if env_name == 'hopper':
        parser.set_defaults(
            reward_scale=0.01,
            rollout=64,
            batch_size=2048,
            learning_rate=0.003,
            lr_schedule_kl_threshold=0.008,
            # learning_rate=0.03,
            # lr_schedule='constant',
        )
    if env_name == 'doublependulum':
        parser.set_defaults(
            reward_scale=0.01,
            learning_rate=0.003,
            lr_schedule='constant',
        )
    if env_name == 'reacher':
        parser.set_defaults(
            reward_scale=0.01,
            learning_rate=0.003,
            rollout=64,
            batch_size=2048,
            # lr_schedule_kl_threshold=0.008,
        )
    if env_name == 'swimmer':
        parser.set_defaults(
            reward_scale=0.1,
            learning_rate=3e-3,
            lr_schedule_kl_threshold=0.008,
            batch_size=1024,
            rollout=64,
            lr_schedule='constant',
        )
    if env_name == 'walker':
        parser.set_defaults(
            reward_scale=0.01,
            learning_rate=0.0003,
            lr_schedule_kl_threshold=0.08,
            rollout=64,
            # lr_schedule='constant',
        )


# noinspection PyUnusedLocal
def add_mujoco_env_args(env, parser):
    p = parser
