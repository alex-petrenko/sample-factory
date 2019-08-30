import gym

from algorithms.memento.mem_wrapper import MemActionSpace
from utils.utils import log


class ObsMemWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = env.observation_space
            self.dict_obs = True
        else:
            self.observation_space = gym.spaces.Dict(spaces=dict(obs=env.observation_space))
            self.dict_obs = False

        self.observation_space.spaces['obs_mem'] = self.observation_space.spaces['obs']

        # two new action spaces: (noop, add), (noop, scroll up, scroll down)
        self.memory_action_spaces = [
            MemActionSpace(2, prior_probs=[0.9, 0.1]),
            MemActionSpace(3, prior_probs=[0.8, 0.1, 0.1]),
        ]

        # modify the original action space
        if isinstance(env.action_space, gym.spaces.Tuple):
            self.action_space = gym.spaces.Tuple(list(env.action_space.spaces) + self.memory_action_spaces)
        else:
            self.action_space = gym.spaces.Tuple([env.action_space] + self.memory_action_spaces)

        self.memory = None
        self.memory_img = None

        self.mem_idx = None
        self.last_observation = None
        self.last_observation_img = None
        self.reset_memory()

    def reset_memory(self):
        self.last_observation = None
        self.last_observation_img = None
        self.memory = []
        self.memory_img = []
        self.mem_idx = 0

    def render(self, mode='human', **kwargs):
        import cv2
        import numpy as np
        # img = np.transpose(self.memory_img[self.mem_idx], (1, 2, 0))
        # img *= 32
        # img = cv2.resize(img, (400, 400))
        cv2.imshow('mem', self.memory_img[self.mem_idx])
        cv2.waitKey(1)

        self.env.render(mode, **kwargs)

    def img_obs(self, obs):
        if self.dict_obs:
            return obs['obs']
        else:
            return obs

    def update_memory(self, mem_actions):
        write_action, scroll_action = mem_actions
        if write_action == 1:
            # writing to memory
            self.memory.append(self.img_obs(self.last_observation))
            self.memory_img.append(self.last_observation_img)
            log.debug('Write to memory')

        if scroll_action == 1:
            # scroll to newer memories
            self.mem_idx += 1
            log.debug('Newer memory %d', self.mem_idx)
        elif scroll_action == 2:
            self.mem_idx -= 1
            log.debug('Older memory %d', self.mem_idx)

        self.mem_idx = max(0, self.mem_idx)
        self.mem_idx = min(0, len(self.memory) - 1)

    def observation(self, obs):
        if self.dict_obs:
            obs['obs_mem'] = self.memory[self.mem_idx]
        else:
            obs = dict(obs=obs, obs_mem=self.memory[self.mem_idx])

        return obs

    def reset(self):
        self.reset_memory()
        obs = self.env.reset()

        self.last_observation = obs
        self.last_observation_img = self.unwrapped.render('rgb_array')
        self.memory.append(self.img_obs(obs))
        self.memory_img.append(self.last_observation_img)
        return self.observation(obs)

    def step(self, actions):
        num_mem_actions = len(self.memory_action_spaces)
        env_actions, memory_actions = actions[:-num_mem_actions], actions[-num_mem_actions:]
        assert len(memory_actions) == num_mem_actions

        self.update_memory(memory_actions)

        if len(env_actions) == 1:
            env_actions = env_actions[0]

        observation, reward, done, info = self.env.step(env_actions)
        self.last_observation = observation
        self.last_observation_img = self.unwrapped.render('rgb_array')
        return self.observation(observation), reward, done, info
