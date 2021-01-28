import gym
from voxel_env.voxel_env_gym import VoxelEnv, make_env_multitask

from envs.env_utils import RewardShapingInterface
from utils.utils import str2bool, log


class Wrapper(gym.Wrapper, RewardShapingInterface):
    """Sets interface for PBT reward shaping, and also extra summaries for multi-task learning."""

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        RewardShapingInterface.__init__(self)

        self.num_agents = env.unwrapped.num_agents
        self.is_multiagent = env.unwrapped.is_multiagent

        # save a reference to this wrapper in the actual env class, for other wrappers and for outside access
        self.env.unwrapped.reward_shaping_interface = self

        self.episode_rewards = [0] * self.num_agents

    def get_default_reward_shaping(self):
        return self.env.unwrapped.get_default_reward_shaping()

    def get_current_reward_shaping(self, agent_idx: int):
        return self.env.unwrapped.get_current_reward_shaping(agent_idx)

    def set_reward_shaping(self, reward_shaping: dict, agent_idx: int):
        return self.env.unwrapped.set_reward_shaping(reward_shaping, agent_idx)

    def reset(self):
        self.episode_rewards = [0] * self.num_agents
        return self.env.reset()

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)

        for i, info in enumerate(infos):
            self.episode_rewards[i] += rewards[i]

            if dones[i]:
                if 'episode_extra_stats' not in info:
                    info['episode_extra_stats'] = dict()
                info['episode_extra_stats'][f'z_{self.env.unwrapped.scenario_name.casefold()}_true_reward'] = info['true_reward']
                info['episode_extra_stats'][f'z_{self.env.unwrapped.scenario_name.casefold()}_reward'] = self.episode_rewards[i]
                self.episode_rewards = [0] * self.num_agents

        return obs, rewards, dones, infos

    def close(self):
        # remove the reference to avoid dependency cycles
        self.env.unwrapped.reward_shaping_interface = None
        return self.env.close()


def make_voxel_env(env_name, cfg=None, env_config=None, **kwargs):
    scenario_name = env_name.split('voxel_env_')[-1]
    log.debug('Using scenario %s', scenario_name)

    if scenario_name.casefold() == 'multitask':
        if env_config is not None and 'worker_index' in env_config:
            task_idx = env_config['worker_index']
        else:
            log.error('Could not find information about task id. Use task_id=0')
            task_idx = 0

        env = make_env_multitask(
            task_idx,
            num_envs=cfg.voxel_num_envs_per_instance,
            num_agents_per_env=cfg.voxel_num_agents_per_env,
            num_simulation_threads=cfg.voxel_num_simulation_threads,
            use_vulkan=cfg.voxel_use_vulkan,
        )
    else:
        env = VoxelEnv(
            scenario_name=scenario_name,
            num_envs=cfg.voxel_num_envs_per_instance,
            num_agents_per_env=cfg.voxel_num_agents_per_env,
            num_simulation_threads=cfg.voxel_num_simulation_threads,
            use_vulkan=cfg.voxel_use_vulkan,
        )

    env = Wrapper(env)

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
    p.add_argument('--voxel_use_vulkan', default=True, type=str2bool, help='Whether to use Vulkan renderer')
