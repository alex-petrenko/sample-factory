from sample_factory.utils.utils import str2bool


def megaverse_override_defaults(env, parser):
    """RL params specific to Megaverse envs."""
    parser.set_defaults(
        encoder_type="conv",
        encoder_subtype="convnet_simple",
        hidden_size=512,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        actor_worker_gpus=[0],
        env_gpu_observations=False,
        exploration_loss="symmetric_kl",
        exploration_loss_coeff=0.001,
    )


def add_megaverse_args(env, parser):
    p = parser
    p.add_argument(
        "--megaverse_num_envs_per_instance", default=1, type=int, help="Num simulated envs per instance of Megaverse"
    )
    p.add_argument(
        "--megaverse_num_agents_per_env",
        default=4,
        type=int,
        help="Number of agents in a single env withing a Megaverse instance. Total number of agents in one Megaverse = num_envs_per_instance * num_agents_per_env",
    )
    p.add_argument(
        "--megaverse_num_simulation_threads",
        default=1,
        type=int,
        help="Number of CPU threads to use per instance of Megaverse",
    )
    p.add_argument("--megaverse_use_vulkan", default=True, type=str2bool, help="Whether to use Vulkan renderer")

    # Team Spirit options
    p.add_argument(
        "--megaverse_increase_team_spirit",
        default=False,
        type=str2bool,
        help="Increase team spirit from 0 to 1 over max_team_spirit_steps during training. At 1, the reward will be completely selfless.",
    )
    p.add_argument(
        "--megaverse_max_team_spirit_steps",
        default=1e9,
        type=float,
        help="Number of training steps when team spirit will hit 1.",
    )
