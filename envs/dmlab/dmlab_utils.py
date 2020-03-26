import os
import os
import shutil
import time
from os.path import join

import cv2
import deepmind_lab
import gym
import numpy as np
from gym.utils import seeding

from envs.dmlab.dmlab30 import LEVEL_MAPPING, dmlab30_num_envs
from envs.dmlab.wrappers.reward_shaping import DmlabRewardShapingWrapper
from envs.env_wrappers import PixelFormatChwWrapper, RecordingWrapper
from utils.utils import project_root, ensure_dir_exists, log

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


class LevelCache:
    def __init__(self, cache_dir):
        ensure_dir_exists(cache_dir)
        self._cache_dir = cache_dir

    def fetch(self, key, pk3_path):
        path = join(self._cache_dir, key)

        if os.path.isfile(path):
            # copy the cached file to the path expected by DeepMind Lab
            shutil.copyfile(path, pk3_path)
            return True

        return False

    def write(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)

        if not os.path.isfile(path):
            # copy the cached file DeepMind Lab has written to the cache directory
            shutil.copyfile(pk3_path, path)


level_cache = LevelCache(join(project_root(), '.dmlab_cache'))


class DmlabGymEnv(gym.Env):
    def __init__(self, level, action_repeat, res_w, res_h, benchmark_mode, renderer, gpu_index, extra_cfg=None):
        self._width = res_w
        self._height = res_h
        self._main_observation = 'DEBUG.CAMERA_INTERLEAVED.PLAYER_VIEW_NO_RETICLE'
        self._action_repeat = action_repeat

        self._random_state = None

        self.level = level
        self.level_name = level.split('/')[-1]

        observation_format = [self._main_observation, 'DEBUG.POS.TRANS']
        config = {'width': self._width, 'height': self._height, 'gpuDeviceIndex': str(gpu_index)}
        if extra_cfg is not None:
            config.update(extra_cfg)
        config = {k: str(v) for k, v in config.items()}

        self._dmlab = deepmind_lab.Lab(
            level, observation_format, config=config, renderer=renderer, level_cache=level_cache,
        )

        self._action_set = ACTION_SET
        self._action_list = np.array(self._action_set, dtype=np.intc)  # DMLAB requires intc type for actions

        self._last_observation = None

        self._render_scale = 5
        self._render_fps = 30
        self._last_frame = time.time()

        self.action_space = gym.spaces.Discrete(len(self._action_set))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)

        self.benchmark_mode = benchmark_mode
        if self.benchmark_mode:
            log.warning('DmLab benchmark mode is true! Use this only for testing, not for actual training runs!')

        self.seed()

    def seed(self, seed=None):
        if self.benchmark_mode:
            initial_seed = 42
        else:
            initial_seed = seeding.hash_seed(seed) % 2 ** 32
        self._random_state = np.random.RandomState(seed=initial_seed)
        return [initial_seed]

    def reset(self):
        self._dmlab.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))
        self._last_observation = self._dmlab.observations()[self._main_observation]
        return self._last_observation

    def step(self, action):
        if self.benchmark_mode:
            # the performance of many DMLab environments heavily depends on what agent is actually doing
            # therefore for purposes of measuring throughput we ignore the actions, this way the agent executes
            # random policy and we can measure raw throughput more precisely
            action = self._random_state.randint(0, self.action_space.n)

        reward = self._dmlab.step(self._action_list[action], num_steps=self._action_repeat)
        done = not self._dmlab.is_running()
        if not done:
            self._last_observation = self._dmlab.observations()[self._main_observation]

        info = {'num_frames': self._action_repeat}
        return self._last_observation, reward, done, info

    def render(self, mode='human'):
        if self._last_observation is None and self._dmlab.is_running():
            self._last_observation = self._dmlab.observations()[self._main_observation]

        img = self._last_observation
        if mode == 'rgb_array':
            return img
        elif mode != 'human':
            raise Exception(f'Rendering mode {mode} not supported')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        scale = self._render_scale
        img_big = cv2.resize(img, (self._width * scale, self._height * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('dmlab', img_big)

        since_last_frame = time.time() - self._last_frame
        wait_time_sec = max(1.0 / self._render_fps - since_last_frame, 0.001)
        wait_time_ms = max(int(1000 * wait_time_sec), 1)
        cv2.waitKey(wait_time_ms)
        self._last_frame = time.time()

    def close(self):
        self._dmlab.close()


class DmLabSpec:
    def __init__(self, name, level, extra_cfg=None):
        self.name = name
        self.level = level
        self.extra_cfg = {} if extra_cfg is None else extra_cfg


DMLAB_ENVS = [
    DmLabSpec('dmlab_benchmark', 'contributed/dmlab30/rooms_collect_good_objects_train'),

    # train a single agent for all 30 DMLab tasks
    DmLabSpec('dmlab_30', None),

    # this is very hard to work with as a benchmark, because FPS fluctuates a lot due to slow resets.
    # also depends a lot on whether levels are in level cache or not
    DmLabSpec('dmlab_benchmark_slow_reset', 'contributed/dmlab30/rooms_keys_doors_puzzle'),

    DmLabSpec('dmlab_sparse', 'contributed/dmlab30/explore_goal_locations_large'),
    DmLabSpec('dmlab_very_sparse', 'contributed/dmlab30/explore_goal_locations_large', extra_cfg={'minGoalDistance': '10'}),
    DmLabSpec('dmlab_sparse_doors', 'contributed/dmlab30/explore_obstructed_goals_large'),
    DmLabSpec('dmlab_nonmatch', 'contributed/dmlab30/rooms_select_nonmatching_object'),
    DmLabSpec('dmlab_watermaze', 'contributed/dmlab30/rooms_watermaze'),
]


def dmlab_env_by_name(name):
    for spec in DMLAB_ENVS:
        if spec.name == name:
            return spec
    raise Exception('Unknown DMLab env')


def get_task_id(env_config):
    if env_config is None:
        return 0
    else:
        num_envs = dmlab30_num_envs()
        return env_config['env_id'] % num_envs


def task_id_to_level(task_id):
    assert 0 <= task_id < dmlab30_num_envs()
    level_name = tuple(LEVEL_MAPPING.keys())[task_id]
    log.debug('Level name %s', level_name)
    return f'contributed/dmlab30/{level_name}'


# noinspection PyUnusedLocal
def make_dmlab_env_impl(spec, cfg, env_config, **kwargs):
    skip_frames = cfg.env_frameskip

    gpu_idx = 0
    if len(cfg.dmlab_gpus) > 0:
        if kwargs.get('env_config') is not None:
            vector_index = kwargs['env_config']['vector_index']
            gpu_idx = cfg.dmlab_gpus[vector_index % len(cfg.dmlab_gpus)]
            log.debug('Using GPU %d for DMLab rendering!', gpu_idx)

    if spec.level is None:
        task_id = get_task_id(env_config)
        level = task_id_to_level(task_id)
    else:
        level = spec.level

    env = DmlabGymEnv(
        level, skip_frames, cfg.res_w, cfg.res_h, cfg.dmlab_throughput_benchmark, cfg.dmlab_renderer,
        gpu_idx, spec.extra_cfg,
    )

    if 'record_to' in cfg and cfg.record_to is not None:
        env = RecordingWrapper(env, cfg.record_to)

    if cfg.pixel_format == 'CHW':
        env = PixelFormatChwWrapper(env)

    env = DmlabRewardShapingWrapper(env)
    return env


def make_dmlab_env(env_name, cfg=None, **kwargs):
    spec = dmlab_env_by_name(env_name)
    return make_dmlab_env_impl(spec, cfg=cfg, **kwargs)

