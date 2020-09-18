from voxel_env.voxel_env_gym import VoxelEnv

from utils.utils import str2bool


def make_voxel_env(env_name, cfg=None, **kwargs):
    env = VoxelEnv(
        num_envs=cfg.voxel_num_envs_per_instance,
        num_agents_per_env=cfg.voxel_num_agents_per_env,
        num_simulation_threads=cfg.voxel_num_simulation_threads,
        vertical_look_limit_rad=cfg.voxel_vertical_look_limit,
        use_vulkan=cfg.voxel_use_vulkan,
    )
    return env


def voxel_env_override_defaults(env, parser):
    """RL params specific to VoxelEnv envs."""
    parser.set_defaults(
        encoder_type='conv',
        encoder_subtype='convnet_simple',
        hidden_size=512,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        actor_worker_gpus=[0],
        exploration_loss='symmetric_kl',
        exploration_loss_coeff=0.001,
    )


def add_voxel_env_args(env, parser):
    p = parser
    p.add_argument('--voxel_num_envs_per_instance', default=1, type=int, help='Num simulated envs per instance of VoxelEnv')
    p.add_argument('--voxel_num_agents_per_env', default=4, type=int, help='Number of agents in a single env withing a VoxelEnv instance. Total number of agents in one VoxelEnv = num_envs_per_instance * num_agents_per_env')
    p.add_argument('--voxel_num_simulation_threads', default=1, type=int, help='Number of CPU threads to use per instance of VoxelEnv')
    p.add_argument('--voxel_vertical_look_limit', default=0.1, type=float, help='Max vertical look in radians')
    p.add_argument('--voxel_use_vulkan', default=False, type=str2bool, help='Whether to use Vulkan renderer')
