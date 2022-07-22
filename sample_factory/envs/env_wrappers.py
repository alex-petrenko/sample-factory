"""
Gym env wrappers that make the environment suitable for the RL algorithms.

"""
import json
import os
from collections import deque
from os.path import join
from typing import Any, Dict, Tuple, Union

import cv2
import gym
import numpy as np

# noinspection PyProtectedMember
from gym import ObservationWrapper, RewardWrapper, spaces
from gym.spaces import Box

from sample_factory.envs.env_utils import num_env_steps
from sample_factory.utils.utils import ensure_dir_exists, log, numpy_all_the_way


def reset_with_info(env):
    """Sometimes we want to get info with the very first frame."""
    obs = env.reset()
    info = {}
    if hasattr(env.unwrapped, "get_info_all"):
        info = env.unwrapped.get_info_all()  # info for the new episode
    return obs, info


def unwrap_env(wrapped_env):
    return wrapped_env.unwrapped


def is_goal_based_env(env):
    dict_obs = isinstance(env.observation_space, spaces.Dict)
    if not dict_obs:
        return False

    for key in ["obs", "goal"]:
        if key not in env.observation_space.spaces:
            return False

    return True


def main_observation_space(env):
    if hasattr(env.observation_space, "spaces"):
        return env.observation_space.spaces["obs"]
    else:
        return env.observation_space


def has_image_observations(observation_space):
    """It's a heuristic."""
    return len(observation_space.shape) >= 2


class StackFramesWrapper(gym.core.Wrapper):
    """
    Gym env wrapper to stack multiple frames.
    Useful for training feed-forward agents on dynamic games.
    """

    def __init__(self, env, stack_past_frames, channel_config="HWC"):
        super(StackFramesWrapper, self).__init__(env)
        if len(env.observation_space.shape) not in [1, 2]:
            raise Exception("Stack frames works with vector observations and 2D single channel images")
        self._stack_past = stack_past_frames
        self._frames = None

        self._image_obs = has_image_observations(env.observation_space)

        self.channel_config = channel_config
        if self._image_obs:
            if self.channel_config == "CHW":
                new_obs_space_shape = (stack_past_frames,) + env.observation_space.shape
            elif self.channel_config == "HWC":
                new_obs_space_shape = env.observation_space.shape + (stack_past_frames,)
            else:
                raise Exception(f"Unknown channel config {self.channel_config}")
        else:
            new_obs_space_shape = list(env.observation_space.shape)
            new_obs_space_shape[0] *= stack_past_frames

        self.observation_space = spaces.Box(
            env.observation_space.low.flat[0],
            env.observation_space.high.flat[0],
            shape=new_obs_space_shape,
            dtype=env.observation_space.dtype,
        )

    def _render_stacked_frames(self):
        if self._image_obs:
            # stack past frames along first dimension
            img = numpy_all_the_way(self._frames)

            if self.channel_config == "CHW":
                return img
            elif self.channel_config == "HWC":
                return np.transpose(img, axes=[1, 2, 0])
            else:
                raise Exception(f"Unknown channel config {self.channel_config}")
        else:
            return np.array(self._frames).flatten()

    def reset(self):
        observation = self.env.reset()
        self._frames = deque([observation] * self._stack_past)
        return self._render_stacked_frames()

    def step(self, action):
        new_observation, reward, done, info = self.env.step(action)
        self._frames.popleft()
        self._frames.append(new_observation)
        return self._render_stacked_frames(), reward, done, info


class SkipFramesWrapper(gym.core.Wrapper):
    """Wrapper for action repeat over N frames to speed up training."""

    def __init__(self, env, skip_frames=4):
        super(SkipFramesWrapper, self).__init__(env)
        self._skip_frames = skip_frames

    def reset(self):
        return self.env.reset()

    def step(self, action):
        done = False
        info = None
        new_observation = None

        total_reward, num_frames = 0, 0
        for i in range(self._skip_frames):
            new_observation, reward, done, info = self.env.step(action)
            num_frames += 1
            total_reward += reward
            if done:
                break

        info["num_frames"] = num_frames
        return new_observation, total_reward, done, info


class SkipAndStackFramesWrapper(StackFramesWrapper):
    """Wrapper for action repeat + stack multiple frames to capture dynamics."""

    def __init__(self, env, skip_frames=4, stack_frames=4, channel_config="HWC"):
        super().__init__(env, stack_past_frames=stack_frames, channel_config=channel_config)
        self._skip_frames = skip_frames

    def step(self, action):
        done = False
        info = {}
        total_reward, num_frames = 0, 0
        for i in range(self._skip_frames):
            new_observation, reward, done, info = self.env.step(action)
            num_frames += 1
            total_reward += reward
            self._frames.popleft()
            self._frames.append(new_observation)
            if done:
                break

        info["num_frames"] = num_frames
        return self._render_stacked_frames(), total_reward, done, info


class NormalizeWrapper(gym.core.Wrapper):
    """
    For environments with vector lowdim input.

    """

    def __init__(self, env):
        super(NormalizeWrapper, self).__init__(env)
        if len(env.observation_space.shape) != 1:
            raise Exception("NormalizeWrapper only works with lowdimensional envs")

        self.wrapped_env = env
        self._normalize_to = 1.0

        self._mean = (env.observation_space.high + env.observation_space.low) * 0.5
        self._max = env.observation_space.high

        self.observation_space = spaces.Box(
            -self._normalize_to,
            self._normalize_to,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def _normalize(self, obs):
        obs -= self._mean
        obs *= self._normalize_to / (self._max - self._mean)
        return obs

    def reset(self):
        observation = self.env.reset()
        return self._normalize(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._normalize(observation), reward, done, info

    @property
    def range(self):
        return [-self._normalize_to, self._normalize_to]


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

    def reset(self):
        return self._observation(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info


class VerticalCropWrapper(ObservationWrapper):
    def __init__(self, env, crop_h):
        # super(VerticalCropWrapper, self).__init__(env)
        super().__init__(env)

        self.crop_h = crop_h
        self.observation_space = self._calc_new_obs_space(env.observation_space)

    def _calc_new_obs_space(self, old_space):
        low, high = old_space.low.flat[0], old_space.high.flat[0]
        h, w, channels = old_space.shape
        new_shape = [self.crop_h, w, channels]
        return spaces.Box(low, high, shape=new_shape, dtype=old_space.dtype)

    # noinspection PyProtectedMember
    def observation(self, observation):
        h = observation.shape[0]
        crop_top = (h - self.crop_h) // 2
        crop_bottom = h - self.crop_h - crop_top
        cropped_obs = observation[crop_top : h - crop_bottom, :, :]
        return cropped_obs


class RewardScalingWrapper(RewardWrapper):
    def __init__(self, env, scaling_factor):
        super(RewardScalingWrapper, self).__init__(env)
        self._scaling = scaling_factor
        self.reward_range = (r * scaling_factor for r in self.reward_range)

    def reward(self, reward):
        return reward * self._scaling


class TimeLimitWrapper(gym.core.Wrapper):
    terminated_by_timer = "TimeLimit.truncated"  # this is to match the default key used by Gym

    def __init__(self, env, limit, random_variation_steps=0):
        super(TimeLimitWrapper, self).__init__(env)
        self._limit = limit
        self._variation_steps = random_variation_steps
        self._num_steps = 0
        self._terminate_in = self._random_limit()

    def _random_limit(self):
        return np.random.randint(-self._variation_steps, self._variation_steps + 1) + self._limit

    def reset(self):
        self._num_steps = 0
        self._terminate_in = self._random_limit()
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if observation is None:
            return observation, reward, done, info

        self._num_steps += num_env_steps([info])
        if done:
            pass
        else:
            if self._num_steps >= self._terminate_in:
                done = True
                info[self.terminated_by_timer] = True

        return observation, reward, done, info


class RemainingTimeWrapper(ObservationWrapper):
    """Designed to be used together with TimeLimitWrapper."""

    def __init__(self, env):
        super(RemainingTimeWrapper, self).__init__(env)

        # adding an additional input dimension to indicate time left before the end of episode
        self.observation_space = spaces.Dict(
            {
                "timer": spaces.Box(0.0, 1.0, shape=[1], dtype=np.float32),
                "obs": env.observation_space,
            }
        )

        wrapped_env = env
        while not isinstance(wrapped_env, TimeLimitWrapper):
            wrapped_env = wrapped_env.env
            if not isinstance(wrapped_env, gym.core.Wrapper):
                raise Exception("RemainingTimeWrapper is supposed to wrap TimeLimitWrapper")
        self.time_limit_wrapper = wrapped_env

    # noinspection PyProtectedMember
    def observation(self, observation):
        num_steps = self.time_limit_wrapper._num_steps
        terminate_in = self.time_limit_wrapper._terminate_in

        dict_obs = {
            "timer": num_steps / terminate_in,
            "obs": observation,
        }
        return dict_obs


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


class ClipRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        reward = min(5.0, reward)
        reward = max(-0.1, reward)
        return reward


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

    def reset(self):
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

        return self.env.reset()

    def _record(self, img):
        frame_name = f"{self._frame_id:05d}.png"
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(join(self._episode_recording_dir, frame_name), img)
        self._frame_id += 1

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

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

        return observation, reward, done, info


GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]


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

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


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

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


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
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


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
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs) -> GymObs:
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


# wrapper from gym, a little modification to get rid of LazyFrame
class FrameStack(gym.ObservationWrapper):
    """Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
          - After :meth:`reset` is called, the frame buffer will be filled with the initial observation. I.e. the observation returned by :meth:`reset` will consist of ``num_stack`-many identical frames,
    Example:
        >>> import gym
        >>> env = gym.make('CarRacing-v1')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 96, 96, 3)
        >>> obs = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)
    """

    def __init__(self, env: gym.Env, num_stack: int, lz4_compress: bool = False):
        """Observation wrapper that stacks the observations in a rolling manner.
        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.
        Args:
            action: The action to step through the environment with
        Returns:
            Stacked observations, reward, done and information from the environment
        """
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.
        Args:
            **kwargs: The kwargs for the environment reset
        Returns:
            The stacked observations
        """
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = None  # Unused
        [self.frames.append(obs) for _ in range(self.num_stack)]

        if kwargs.get("return_info", False):
            return obs, info
        else:
            return obs
