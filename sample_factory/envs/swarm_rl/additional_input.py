import gym
import numpy as np


class QuadsAdditionalInputWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_space = gym.spaces.Box(-5.0, 5.0, shape=(21, ))
        self.observation_space = obs_space

    def _modify_obs(self, obs):
        # TODO: this is test code, if this works it should be moved to the env implementation
        pos = obs[0:3]
        obs[0:3] = np.clip(pos, -10.0, 10.0)

        vel = obs[3:6]
        obs[3:6] = np.clip(vel, -5.0, 5.0)

        omega = obs[-3:]
        obs[-3:] = np.clip(omega, -5.0, 5.0)

        dist_to_goal = obs[:3]
        dist_to_goal_scaled = 10.0 * dist_to_goal
        dist_to_goal_scaled = np.clip(dist_to_goal_scaled, -3.0, 3.0)

        obs = np.concatenate((obs, dist_to_goal_scaled))
        return obs

    def reset(self):
        obs = self.env.reset()
        return self._modify_obs(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        # dist_to_goal = obs[:3]
        # dist_to_goal_norm_centimeters = 100 * np.linalg.norm(dist_to_goal)
        # extra_reward = -dist_to_goal_norm_centimeters / 400 + 0.05
        # extra_reward = max(0.0, extra_reward)

        obs = self._modify_obs(obs)
        # log.warning('Min/max state %.2f, %.2f, argmin/argmax %d %d', obs.min(), obs.max(), np.argmin(obs), np.argmax(obs))

        return obs, rew, done, info

    def close(self):
        return self.env.close()
