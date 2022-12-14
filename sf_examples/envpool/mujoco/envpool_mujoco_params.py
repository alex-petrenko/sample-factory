import argparse

from sf_examples.envpool.envpool_utils import add_envpool_common_args


def mujoco_envpool_override_defaults(env: str, parser: argparse.ArgumentParser) -> None:
    # High-throughput parameters optimized for wall-time performance (e.g. getting the highest reward in 10 minutes).
    # See sf_examples/mujoco/mujoco_params.py for more standard parameters similar to SB3/CleanRL that are known
    # to provide good sample efficiency

    parser.set_defaults(
        batched_sampling=True,
        num_workers=1,  # envpool takes care of parallelization, so use only 1 worker?
        num_envs_per_worker=1,  # two envs per worker for double-buffered sampling, one for single-buffered
        worker_num_splits=1,  # change to 2 to enable double-buffered sampling
        train_for_env_steps=10000000,
        encoder_mlp_layers=[256, 128, 64],
        nonlinearity="elu",  # as in https://github.com/Denys88/rl_games/blob/d8645b2678c0d8a6e98a6e3f2b17f0ecfbff71ad/rl_games/configs/mujoco/ant_envpool.yaml#L24
        kl_loss_coeff=0.1,
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization="torch_default",
        reward_scale=0.1,
        rollout=64,
        max_grad_norm=1.0,
        num_epochs=4,
        num_batches_per_epoch=2,
        batch_size=2048,  # 2048 * 2 = 8192 env steps per training iteration
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.0,
        learning_rate=3e-4,  # does not matter because it will be adaptively changed anyway
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.008,
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        recurrence=1,
        normalize_input=True,
        normalize_returns=True,
        value_bootstrap=True,
        experiment_summaries_interval=15,
        save_every_sec=15,
        save_best_every_sec=15,
        async_rl=False,
        serial_mode=True,  # we're running with 1 worker in sync mode, so might as well run everything in one process
    )

    if env in env_configs:
        parser.set_defaults(**env_configs[env])


env_configs = dict(
    mujoco_ant=dict(
        encoder_mlp_layers=[128, 64, 32],
        nonlinearity="elu",
        learning_rate=3e-4,
        rollout=64,
        num_epochs=4,
        num_batches_per_epoch=2,
    ),
    mujoco_halfcheetah=dict(
        encoder_mlp_layers=[128, 64, 32],
        nonlinearity="elu",
        learning_rate=5e-4,
        rollout=256,
        num_epochs=5,
        num_batches_per_epoch=8,
    ),
    mujoco_humanoid=dict(
        encoder_mlp_layers=[512, 256, 128],
        nonlinearity="elu",
        learning_rate=3e-4,
        rollout=128,
        num_epochs=5,
        num_batches_per_epoch=4,
    ),
    mujoco_hopper=dict(
        encoder_mlp_layers=[256, 128, 64],
        nonlinearity="elu",
        learning_rate=5e-4,
        rollout=64,
        num_epochs=5,
        num_batches_per_epoch=2,
    ),
    mujoco_walker=dict(
        encoder_mlp_layers=[256, 128, 64],
        nonlinearity="elu",
        learning_rate=3e-4,
        rollout=128,
        num_epochs=5,
        num_batches_per_epoch=4,
    ),
)


# noinspection PyUnusedLocal
def add_mujoco_envpool_env_args(env, parser, evaluation: bool = False) -> None:
    # in case we need to add more args in the future
    parser.add_argument(
        "--env_agents",
        default=1 if evaluation else 64,
        type=int,
        help="Num agents in each envpool (if used)",
    )

    add_envpool_common_args(env, parser)
