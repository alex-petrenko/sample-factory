"""
Gym env wrappers that make the environment suitable for the RL algorithms.

"""
import json
import os
from os.path import join
from typing import Any, Dict, Tuple, Union

import cv2
import gym
import numpy as np

# noinspection PyProtectedMember
from gym import ObservationWrapper, RewardWrapper, spaces

from sample_factory.envs.env_utils import num_env_steps
from sample_factory.utils.utils import ensure_dir_exists, log


def has_image_observations(observation_space):
    """It's a heuristic."""
    return len(observation_space.shape) >= 2


class ResizeWrapper(gym.core.Wrapper):
    """Resize observation frames to specified (w,h) and convert to grayscale."""

    def __init__(self, env, w, h, grayscale=True, add_channel_dim=False, area_interpolation=False):
        super(ResizeWrapper, self).__init__(env)

        self.w = w
        self.h = h
        self.grayscale = grayscale
        self.add_channel_dim = add_channel_dim
        self.interpolation = cv2.INTER_AREA if area_interpolation else cv2.INTER_NEAREST

        if isinstance(env.observation_space, spaces.Dict):
            # TODO: does this even work?
            new_spaces = {}
            for key, space in env.observation_space.spaces.items():
                new_spaces[key] = self._calc_new_obs_space(space)
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = self._calc_new_obs_space(env.observation_space)

    def _calc_new_obs_space(self, old_space):
        low, high = old_space.low.flat[0], old_space.high.flat[0]

        if self.grayscale:
            new_shape = [self.h, self.w, 1] if self.add_channel_dim else [self.h, self.w]
        else:
            if len(old_space.shape) > 2:
                channels = old_space.shape[-1]
                new_shape = [self.h, self.w, channels]
            else:
                new_shape = [self.h, self.w, 1] if self.add_channel_dim else [self.h, self.w]

        return spaces.Box(low, high, shape=new_shape, dtype=old_space.dtype)

    def _convert_obs(self, obs):
        if obs is None:
            return obs

        obs = cv2.resize(obs, (self.w, self.h), interpolation=self.interpolation)
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        if self.add_channel_dim:
            return obs[:, :, None]  # add new dimension (expected by tensorflow)
        else:
            return obs

    def _observation(self, obs):
        if isinstance(obs, dict):
            new_obs = {}
            for key, value in obs.items():
                new_obs[key] = self._convert_obs(value)
            return new_obs
        else:
            return self._convert_obs(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(obs), reward, terminated, truncated, info


class RewardScalingWrapper(RewardWrapper):
    def __init__(self, env, scaling_factor):
        super(RewardScalingWrapper, self).__init__(env)
        self._scaling = scaling_factor
        self.reward_range = (r * scaling_factor for r in self.reward_range)

    def reward(self, reward):
        return reward * self._scaling


class TimeLimitWrapper(gym.core.Wrapper):
    def __init__(self, env, limit, random_variation_steps=0):
        super(TimeLimitWrapper, self).__init__(env)
        self._limit = limit
        self._variation_steps = random_variation_steps
        self._num_steps = 0
        self._terminate_in = self._random_limit()

    def _random_limit(self):
        return np.random.randint(-self._variation_steps, self._variation_steps + 1) + self._limit

    def reset(self, **kwargs):
        self._num_steps = 0
        self._terminate_in = self._random_limit()
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if observation is None:
            return observation, reward, terminated, truncated, info

        self._num_steps += num_env_steps([info])
        if terminated or truncated:
            pass
        elif self._num_steps >= self._terminate_in:
            truncated = True

        return observation, reward, terminated, truncated, info


class PixelFormatChwWrapper(ObservationWrapper):
    """TODO? This can be optimized for VizDoom, can we query CHW directly from VizDoom?"""

    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            img_obs_space = env.observation_space["obs"]
            self.dict_obs_space = True
        else:
            img_obs_space = env.observation_space
            self.dict_obs_space = False

        if not has_image_observations(img_obs_space):
            raise Exception("Pixel format wrapper only works with image-based envs")

        obs_shape = img_obs_space.shape
        max_num_img_channels = 4

        if len(obs_shape) <= 2:
            raise Exception("Env obs do not have channel dimension?")

        if obs_shape[0] <= max_num_img_channels:
            raise Exception("Env obs already in CHW format?")

        h, w, c = obs_shape
        low, high = img_obs_space.low.flat[0], img_obs_space.high.flat[0]
        new_shape = [c, h, w]

        if self.dict_obs_space:
            dtype = (
                env.observation_space.spaces["obs"].dtype
                if env.observation_space.spaces["obs"].dtype is not None
                else np.float32
            )
        else:
            dtype = env.observation_space.dtype if env.observation_space.dtype is not None else np.float32

        new_img_obs_space = spaces.Box(low, high, shape=new_shape, dtype=dtype)

        if self.dict_obs_space:
            self.observation_space = env.observation_space
            self.observation_space.spaces["obs"] = new_img_obs_space
        else:
            self.observation_space = new_img_obs_space

        self.action_space = env.action_space

    @staticmethod
    def _transpose(obs):
        return np.transpose(obs, (2, 0, 1))  # HWC to CHW for PyTorch

    def observation(self, observation):
        if observation is None:
            return observation

        if self.dict_obs_space:
            observation["obs"] = self._transpose(observation["obs"])
        else:
            observation = self._transpose(observation)
        return observation


class RecordingWrapper(gym.core.Wrapper):
    def __init__(self, env, record_to, player_id):
        super().__init__(env)

        self._record_to = record_to
        self._episode_recording_dir = None
        self._record_id = 0
        self._frame_id = 0
        self._player_id = player_id
        self._recorded_episode_reward = 0
        self._recorded_episode_shaping_reward = 0

        self._recorded_actions = []

        # Experimental! Recording Doom replay. Does not work in all scenarios, e.g. when there are in-game bots.
        self.unwrapped.record_to = record_to

    def reset(self, **kwargs):
        if self._episode_recording_dir is not None and self._record_id > 0:
            # save actions to text file
            with open(join(self._episode_recording_dir, "actions.json"), "w") as actions_file:
                json.dump(self._recorded_actions, actions_file)

            # rename previous episode dir
            reward = self._recorded_episode_reward + self._recorded_episode_shaping_reward
            new_dir_name = self._episode_recording_dir + f"_r{reward:.2f}"
            os.rename(self._episode_recording_dir, new_dir_name)
            log.info(
                "Finished recording %s (rew %.3f, shaping %.3f)",
                new_dir_name,
                reward,
                self._recorded_episode_shaping_reward,
            )

        dir_name = f"ep_{self._record_id:03d}_p{self._player_id}"
        self._episode_recording_dir = join(self._record_to, dir_name)
        ensure_dir_exists(self._episode_recording_dir)

        self._record_id += 1
        self._frame_id = 0
        self._recorded_episode_reward = 0
        self._recorded_episode_shaping_reward = 0

        self._recorded_actions = []

        return self.env.reset(**kwargs)

    def _record(self, img):
        frame_name = f"{self._frame_id:05d}.png"
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(join(self._episode_recording_dir, frame_name), img)
        self._frame_id += 1

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(action, np.ndarray):
            self._recorded_actions.append(action.tolist())
        elif np.issubdtype(type(action), np.integer):
            self._recorded_actions.append(int(action))
        else:
            self._recorded_actions.append(action)

        self._record(observation)
        self._recorded_episode_reward += reward
        if hasattr(self.env.unwrapped, "_total_shaping_reward"):
            # noinspection PyProtectedMember
            self._recorded_episode_shaping_reward = self.env.unwrapped._total_shaping_reward

        return observation, reward, terminated, truncated, info


GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]


# wrapper from CleanRL / Stable Baselines
class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    :param env: the environment to wrap
    :param noop_max: the maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info = {}
        for _ in range(noops):
            obs, rew, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated | truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


# wrapper from CleanRL / Stable Baselines
class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated | truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated | truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


# wrapper from CleanRL / Stable Baselines
class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> GymStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated | truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


# wrapper from CleanRL / Stable Baselines
class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        info = {}
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated | truncated:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[GymObs, Dict]:
        return self.env.reset(**kwargs)


# wrapper from CleanRL / Stable Baselines
class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.
    :param env: the environment
    """

    def __init__(self, env: gym.Env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward: float) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.
        :param reward:
        :return:
        """
        return np.sign(reward)


class NumpyObsWrapper(gym.ObservationWrapper):
    """
    RL algorithm generally expects numpy arrays or Tensors as observations. Atari envs for example return
    LazyFrames which need to be converted to numpy arrays before we actually use them.
    """

    def observation(self, observation: Any) -> np.ndarray:
        return np.array(observation)
