import gym
from voxel_env.voxel_env_gym import VoxelEnv

from envs.env_utils import RewardShapingInterface
from utils.utils import str2bool


class RewardShapingWrapper(gym.Wrapper, RewardShapingInterface):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        RewardShapingInterface.__init__(self)

        self.num_agents = env.num_agents
        self.is_multiagent = env.is_multiagent

        # save a reference to this wrapper in the actual env class, for other wrappers and for outside access
        self.env.unwrapped.reward_shaping_interface = self

    def get_default_reward_shaping(self):
        return self.env.get_default_reward_shaping()

    def get_current_reward_shaping(self, agent_idx: int):
        return self.env.get_current_reward_shaping(agent_idx)

    def set_reward_shaping(self, reward_shaping: dict, agent_idx: int):
        return self.env.set_reward_shaping(reward_shaping, agent_idx)

    def close(self):
        # remove the reference to avoid dependency cycles
        self.env.unwrapped.reward_shaping_interface = None
        return self.env.close()


def make_voxel_env(env_name, cfg=None, **kwargs):
    env = VoxelEnv(
        num_envs=cfg.voxel_num_envs_per_instance,
        num_agents_per_env=cfg.voxel_num_agents_per_env,
        num_simulation_threads=cfg.voxel_num_simulation_threads,
        vertical_look_limit_rad=cfg.voxel_vertical_look_limit,
        use_vulkan=cfg.voxel_use_vulkan,
    )

    env = RewardShapingWrapper(env)
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
