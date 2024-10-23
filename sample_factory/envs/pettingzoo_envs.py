"""
Gym env wrappers for PettingZoo -> Gymnasium transition.
"""

import gymnasium as gym


class PettingZooParallelEnv(gym.Env):
    def __init__(self, env):
        if not all_equal([env.observation_space(a) for a in env.possible_agents]):
            raise ValueError("All observation spaces must be equal")

        if not all_equal([env.action_space(a) for a in env.possible_agents]):
            raise ValueError("All action spaces must be equal")

        self.env = env
        self.metadata = env.metadata
        self.render_mode = env.render_mode if hasattr(env, "render_mode") else env.unwrapped.render_mode
        self.observation_space = normalize_observation_space(env.observation_space(env.possible_agents[0]))
        self.action_space = env.action_space(env.possible_agents[0])
        self.num_agents = env.max_num_agents
        self.is_multiagent = True

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        obs = [normalize_observation(obs.get(a)) for a in self.env.possible_agents]
        infos = [infos[a] if a in infos else dict(is_active=False) for a in self.env.possible_agents]
        return obs, infos

    def step(self, actions):
        actions = dict(zip(self.env.possible_agents, actions))
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        if not self.env.agents:
            obs, infos = self.env.reset()

        obs = [normalize_observation(obs.get(a)) for a in self.env.possible_agents]
        rewards = [rewards.get(a) for a in self.env.possible_agents]
        terminations = [terminations.get(a) for a in self.env.possible_agents]
        truncations = [truncations.get(a) for a in self.env.possible_agents]
        infos = [normalize_info(infos[a], a) if a in infos else dict(is_active=False) for a in self.env.possible_agents]
        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def all_equal(l_) -> bool:
    return all(v == l_[0] for v in l_)


def normalize_observation_space(obs_space):
    """Normalize observation space with the key "obs" that's specially handled as the main value."""
    if isinstance(obs_space, gym.spaces.Dict) and "observation" in obs_space.spaces:
        spaces = dict(obs_space.spaces)
        spaces["obs"] = spaces["observation"]
        del spaces["observation"]
        obs_space = gym.spaces.Dict(spaces)

    return obs_space


def normalize_observation(obs):
    if isinstance(obs, dict) and "observation" in obs:
        obs["obs"] = obs["observation"]
        del obs["observation"]

    return obs


def normalize_info(info, agent):
    """active_agent is available when using `turn_based_aec_to_parallel` of PettingZoo."""
    if isinstance(info, dict) and "active_agent" in info:
        info["is_active"] = info["active_agent"] == agent

    return info
