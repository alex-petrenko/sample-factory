import os
import time
import random
import shutil
from os.path import join

import cv2
import deepmind_lab
import gym
import numpy as np

from sample_factory.envs.dmlab import dmlab_level_cache
from sample_factory.envs.dmlab.dmlab30 import DMLAB_INSTRUCTIONS, DMLAB_MAX_INSTRUCTION_LEN, DMLAB_VOCABULARY_SIZE
from sample_factory.envs.dmlab.dmlab_utils import string_to_hash_bucket
from sample_factory.utils.utils import log, ensure_dir_exists

ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)


EXTENDED_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-10, 0, 0, 0, 0, 0, 0),  # Small Look Left
    (10, 0, 0, 0, 0, 0, 0),   # Small Look Right
    (-60, 0, 0, 0, 0, 0, 0),  # Large Look Left
    (60, 0, 0, 0, 0, 0, 0),   # Large Look Right
    (0, 10, 0, 0, 0, 0, 0),   # Look Down
    (0, -10, 0, 0, 0, 0, 0),  # Look Up
    (-10, 0, 0, 1, 0, 0, 0),  # Forward + Small Look Left
    (10, 0, 0, 1, 0, 0, 0),   # Forward + Small Look Right
    (-60, 0, 0, 1, 0, 0, 0),  # Forward + Large Look Left
    (60, 0, 0, 1, 0, 0, 0),   # Forward + Large Look Right
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)


def dmlab_level_to_level_name(level):
    level_name = level.split('/')[-1]
    return level_name


class DmlabGymEnv(gym.Env):
    def __init__(
            self, task_id, level, action_repeat, res_w, res_h, benchmark_mode, renderer, dataset_path,
            with_instructions, extended_action_set, use_level_cache, level_cache_path,
            gpu_index, extra_cfg=None,
    ):
        self.width = res_w
        self.height = res_h

        # self._main_observation = 'DEBUG.CAMERA_INTERLEAVED.PLAYER_VIEW_NO_RETICLE'
        self.main_observation = 'RGB_INTERLEAVED'
        self.instructions_observation = DMLAB_INSTRUCTIONS
        self.with_instructions = with_instructions and not benchmark_mode

        self.action_repeat = action_repeat

        self.random_state = None

        self.task_id = task_id
        self.level = level
        self.level_name = dmlab_level_to_level_name(self.level)

        # the policy index which currently acts in the environment
        self.curr_policy_idx = 0
        self.curr_cache = dmlab_level_cache.DMLAB_GLOBAL_LEVEL_CACHE[self.curr_policy_idx]

        self.instructions = np.zeros([DMLAB_MAX_INSTRUCTION_LEN], dtype=np.int32)

        observation_format = [self.main_observation]
        if self.with_instructions:
            observation_format += [self.instructions_observation]

        config = {
            'width': self.width,
            'height': self.height,
            'gpuDeviceIndex': str(gpu_index),
            'datasetPath': dataset_path,
        }

        if extra_cfg is not None:
            config.update(extra_cfg)
        config = {k: str(v) for k, v in config.items()}

        self.use_level_cache = use_level_cache
        self.level_cache_path = ensure_dir_exists(level_cache_path)

        env_level_cache = self if use_level_cache else None
        self.env_uses_level_cache = False  # will be set to True when this env instance queries the cache
        self.last_reset_seed = None

        if env_level_cache is not None:
            if not isinstance(self.curr_cache, dmlab_level_cache.DmlabLevelCacheGlobal):
                raise Exception(
                    'DMLab global level cache object is not initialized! Make sure to call'
                    'dmlab_ensure_global_cache_initialized() in the main thread before you fork any child processes'
                    'or create any DMLab envs'
                )

        self.dmlab = deepmind_lab.Lab(
            level, observation_format, config=config, renderer=renderer, level_cache=env_level_cache,
        )

        self.action_set = EXTENDED_ACTION_SET if extended_action_set else ACTION_SET
        self.action_list = np.array(self.action_set, dtype=np.intc)  # DMLAB requires intc type for actions

        self.last_observation = None

        self.render_scale = 5
        self.render_fps = 30
        self.last_frame = time.time()

        self.action_space = gym.spaces.Discrete(len(self.action_set))

        self.observation_space = gym.spaces.Dict(
            obs=gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        )
        if self.with_instructions:
            self.observation_space.spaces[self.instructions_observation] = gym.spaces.Box(
                low=0, high=DMLAB_VOCABULARY_SIZE, shape=[DMLAB_MAX_INSTRUCTION_LEN], dtype=np.int32,
            )

        self.benchmark_mode = benchmark_mode
        if self.benchmark_mode:
            log.warning('DmLab benchmark mode is true! Use this only for testing, not for actual training runs!')

        self.seed()

    def seed(self, seed=None):
        if self.benchmark_mode:
            initial_seed = 42
        else:
            if seed is None:
                initial_seed = random.randint(0, int(1e9))
            else:
                initial_seed = seed

        self.random_state = np.random.RandomState(seed=initial_seed)
        return [initial_seed]

    def format_obs_dict(self, env_obs_dict):
        """SampleFactory traditionally uses 'obs' key for the 'main' observation."""
        env_obs_dict['obs'] = env_obs_dict.pop(self.main_observation)

        instr = env_obs_dict.get(self.instructions_observation)
        self.instructions[:] = 0
        if instr is not None:
            instr_words = instr.split()
            for i, word in enumerate(instr_words):
                self.instructions[i] = string_to_hash_bucket(word, DMLAB_VOCABULARY_SIZE)

            env_obs_dict[self.instructions_observation] = self.instructions

        return env_obs_dict

    def reset(self):
        if self.use_level_cache:
            self.curr_cache = dmlab_level_cache.DMLAB_GLOBAL_LEVEL_CACHE[self.curr_policy_idx]
            self.last_reset_seed = self.curr_cache.get_unused_seed(self.level, self.random_state)
        else:
            self.last_reset_seed = self.random_state.randint(0, 2 ** 31 - 1)

        self.dmlab.reset(seed=self.last_reset_seed)
        self.last_observation = self.format_obs_dict(self.dmlab.observations())
        return self.last_observation

    def step(self, action):
        if self.benchmark_mode:
            # the performance of many DMLab environments heavily depends on what agent is actually doing
            # therefore for purposes of measuring throughput we ignore the actions, this way the agent executes
            # random policy and we can measure raw throughput more precisely
            action = random.randrange(0, self.action_space.n)

        reward = self.dmlab.step(self.action_list[action], num_steps=self.action_repeat)
        done = not self.dmlab.is_running()

        if not done:
            obs_dict = self.format_obs_dict(self.dmlab.observations())
            self.last_observation = obs_dict

        info = {'num_frames': self.action_repeat}
        return self.last_observation, reward, done, info

    def render(self, mode='human'):
        if self.last_observation is None and self.dmlab.is_running():
            self.last_observation = self.dmlab.observations()

        img = self.last_observation[self.main_observation]
        if mode == 'rgb_array':
            return img
        elif mode != 'human':
            raise Exception(f'Rendering mode {mode} not supported')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        scale = self.render_scale
        img_big = cv2.resize(img, (self.width * scale, self.height * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('dmlab', img_big)

        since_last_frame = time.time() - self.last_frame
        wait_time_sec = max(1.0 / self.render_fps - since_last_frame, 0.001)
        wait_time_ms = max(int(1000 * wait_time_sec), 1)
        cv2.waitKey(wait_time_ms)
        self.last_frame = time.time()

    def close(self):
        self.dmlab.close()

    def fetch(self, key, pk3_path):
        """Environment object itself acts as a proxy to the global level cache."""
        if not self.env_uses_level_cache:
            self.env_uses_level_cache = True
            # log.debug('Env %s uses level cache!', self.level_name)

        path = join(self.level_cache_path, key)

        if os.path.isfile(path):
            # copy the cached file to the path expected by DeepMind Lab
            shutil.copyfile(path, pk3_path)
            return True
        else:
            log.warning('Cache miss in environment %s key: %s!', self.level_name, key)
            return False

    def write(self, key, pk3_path):
        """Environment object itself acts as a proxy to the global level cache."""
        log.debug('Add new level to cache! Level %s seed %r key %s', self.level_name, self.last_reset_seed, key)
        self.curr_cache.add_new_level(self.level, self.last_reset_seed, key, pk3_path)
