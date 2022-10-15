import gym
from megaverse.megaverse_env import MegaverseEnv, make_env_multitask

from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface
from sample_factory.utils.utils import log


class MegaverseSpec:
    def __init__(self, name):
        self.name = name


MEGAVERSE_ENVS = [
    MegaverseSpec("megaverse_TowerBuilding"),
]


class Wrapper(gym.Wrapper, RewardShapingInterface, TrainingInfoInterface):
    """Sets interface for PBT reward shaping, and also extra summaries for multi-task learning."""

    def __init__(self, env, increase_team_spirit, max_team_spirit_steps):
        gym.Wrapper.__init__(self, env)
        RewardShapingInterface.__init__(self)
        TrainingInfoInterface.__init__(self)

        self.num_agents = env.unwrapped.num_agents
        self.is_multiagent = env.unwrapped.is_multiagent

        self.episode_rewards = [0] * self.num_agents

        self.increase_team_spirit = increase_team_spirit
        self.max_team_spirit_steps = max_team_spirit_steps

        self.approx_total_training_steps = 0

    def get_default_reward_shaping(self):
        return self.env.unwrapped.get_default_reward_shaping()

    def get_current_reward_shaping(self, agent_idx: int):
        return self.env.unwrapped.get_current_reward_shaping(agent_idx)

    def set_reward_shaping(self, reward_shaping: dict, agent_idx: int):
        return self.env.unwrapped.set_reward_shaping(reward_shaping, agent_idx)

    def reset(self, **kwargs):
        self.episode_rewards = [0] * self.num_agents
        return self.env.reset(), {}

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)

        for i, info in enumerate(infos):
            self.episode_rewards[i] += rewards[i]

            if dones[i]:
                if "episode_extra_stats" not in info:
                    info["episode_extra_stats"] = dict()
                extra_stats = info["episode_extra_stats"]
                extra_stats[f"z_{self.env.unwrapped.scenario_name.casefold()}_true_reward"] = info["true_reward"]
                extra_stats[f"z_{self.env.unwrapped.scenario_name.casefold()}_reward"] = self.episode_rewards[i]

                approx_total_training_steps = self.training_info.get("approx_total_training_steps", 0)
                extra_stats["z_approx_total_training_steps"] = approx_total_training_steps

                self.episode_rewards[i] = 0

                if self.increase_team_spirit:
                    rew_shaping = self.get_current_reward_shaping(i)
                    rew_shaping["teamSpirit"] = min(approx_total_training_steps / self.max_team_spirit_steps, 1.0)
                    self.set_reward_shaping(rew_shaping, i)
                    extra_stats["teamSpirit"] = rew_shaping["teamSpirit"]

        terminated = dones
        truncated = dones
        return obs, rewards, terminated, truncated, infos


def make_megaverse(env_name, cfg=None, env_config=None, **kwargs):
    scenario_name = env_name.split("megaverse_")[-1].casefold()
    log.debug("Using scenario %s", scenario_name)

    if "multitask" in scenario_name:
        if env_config is not None and "worker_index" in env_config:
            task_idx = env_config["worker_index"]
        else:
            log.warning(
                "Could not find information about task id. Use task_id=0. (It is okay if this message appears once)"
            )
            task_idx = 0

        env = make_env_multitask(
            scenario_name,
            task_idx,
            num_envs=cfg.megaverse_num_envs_per_instance,
            num_agents_per_env=cfg.megaverse_num_agents_per_env,
            num_simulation_threads=cfg.megaverse_num_simulation_threads,
            use_vulkan=cfg.megaverse_use_vulkan,
        )
    else:
        env = MegaverseEnv(
            scenario_name=scenario_name,
            num_envs=cfg.megaverse_num_envs_per_instance,
            num_agents_per_env=cfg.megaverse_num_agents_per_env,
            num_simulation_threads=cfg.megaverse_num_simulation_threads,
            use_vulkan=cfg.megaverse_use_vulkan,
        )

    env = Wrapper(env, cfg.megaverse_increase_team_spirit, cfg.megaverse_max_team_spirit_steps)
    return env
