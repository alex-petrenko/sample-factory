from __future__ import annotations

import multiprocessing
import os
import pickle
from dataclasses import dataclass
from os.path import join
from typing import Dict, List, NamedTuple, Optional

import gymnasium as gym

from sample_factory.algo.utils.action_distributions import calc_num_actions
from sample_factory.algo.utils.context import set_global_context, sf_global_context
from sample_factory.algo.utils.make_env import BatchedVecEnv, NonBatchedVecEnv, make_env_func_batched
from sample_factory.envs.env_utils import get_default_reward_shaping
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log, project_tmp_dir

ENV_INFO_PROTOCOL_VERSION = 1


class EnvSpaces(NamedTuple):
    obs_space: gym.Space
    action_space: gym.Space
    num_agents: int


class EnvExecutionInfo(NamedTuple):
    gpu_actions: bool
    gpu_observations: bool


class EnvActionInfo(NamedTuple):
    action_splits: Optional[List[int]]
    all_discrete: Optional[bool]


@dataclass(init=False)
class EnvInfo:
    spaces: EnvSpaces
    execution: EnvExecutionInfo
    action_info: EnvActionInfo
    frameskip: int
    reward_shaping_scheme: Optional[Dict[str, float]] = None
    env_info_protocol_version: Optional[int] = None

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_agents: int,
        gpu_actions: bool,
        gpu_observations: bool,
        action_splits: Optional[List[int]],
        all_discrete: Optional[bool],
        frameskip: int,
        reward_shaping_scheme: Optional[Dict[str, float]] = None,
        env_info_protocol_version: Optional[int] = None,
    ):
        self.spaces = EnvSpaces(obs_space, action_space, num_agents)
        self.execution = EnvExecutionInfo(gpu_actions, gpu_observations)
        self.action_info = EnvActionInfo(action_splits, all_discrete)
        self.frameskip = frameskip
        self.reward_shaping_scheme = reward_shaping_scheme
        self.env_info_protocol_version = env_info_protocol_version

    @property
    def obs_space(self) -> gym.Space:
        return self.spaces.obs_space

    @property
    def action_space(self) -> gym.Space:
        return self.spaces.action_space

    @property
    def num_agents(self) -> int:
        return self.spaces.num_agents

    @property
    def gpu_actions(self) -> bool:
        return self.execution.gpu_actions

    @property
    def gpu_observations(self) -> bool:
        return self.execution.gpu_observations

    @property
    def action_splits(self) -> Optional[List[int]]:
        return self.action_info.action_splits

    @property
    def all_discrete(self) -> Optional[bool]:
        return self.action_info.all_discrete


def extract_env_info(env: BatchedVecEnv | NonBatchedVecEnv, cfg: Config) -> EnvInfo:
    obs_space = env.observation_space
    action_space = env.action_space
    num_agents = env.num_agents

    gpu_actions = cfg.env_gpu_actions
    gpu_observations = cfg.env_gpu_observations

    frameskip = cfg.env_frameskip

    reward_shaping_scheme = get_default_reward_shaping(env)

    action_splits = None
    all_discrete = None
    if isinstance(action_space, gym.spaces.Tuple):
        action_splits = [calc_num_actions(space) for space in action_space]
        all_discrete = all(isinstance(space, gym.spaces.Discrete) for space in action_space)

    env_info = EnvInfo(
        obs_space=obs_space,
        action_space=action_space,
        num_agents=num_agents,
        gpu_actions=gpu_actions,
        gpu_observations=gpu_observations,
        action_splits=action_splits,
        all_discrete=all_discrete,
        frameskip=frameskip,
        reward_shaping_scheme=reward_shaping_scheme,
        env_info_protocol_version=ENV_INFO_PROTOCOL_VERSION,
    )
    return env_info


def check_env_info(env: BatchedVecEnv | NonBatchedVecEnv, env_info: EnvInfo, cfg: Config) -> None:
    new_env_info = extract_env_info(env, cfg)
    if new_env_info != env_info:
        cache_filename = env_info_cache_filename(cfg)
        log.error(
            f"Env info does not match the cached value: {env_info} != {new_env_info}. Deleting the cache entry {cache_filename}"
        )

        try:
            os.remove(cache_filename)
        except OSError:
            # ignoring errors, this is not super important
            pass

        log.error(
            "This is likely because the environment has changed after the cache entry was created. "
            "Either restart the experiment to fix this or run with --use_env_info_cache=False to avoid such problems in the future."
        )
        raise ValueError("Env info mismatch. See logs above for details.")


def spawn_tmp_env_and_get_info(sf_context, res_queue, cfg):
    set_global_context(sf_context)

    tmp_env = make_env_func_batched(cfg, env_config=None)
    env_info = extract_env_info(tmp_env, cfg)
    tmp_env.close()
    del tmp_env

    log.debug("Env info: %r", env_info)
    res_queue.put(env_info)


def env_info_cache_filename(cfg: Config) -> str:
    return join(project_tmp_dir(), f"env_info_{cfg.env}")


def obtain_env_info_in_a_separate_process(cfg: Config) -> EnvInfo:
    cache_filename = env_info_cache_filename(cfg)
    if cfg.use_env_info_cache and os.path.isfile(cache_filename):
        log.debug(f"Loading env info from cache: {cache_filename}")
        with open(cache_filename, "rb") as fobj:
            env_info = pickle.load(fobj)
            if env_info.env_info_protocol_version == ENV_INFO_PROTOCOL_VERSION:
                return env_info

    sf_context = sf_global_context()

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=spawn_tmp_env_and_get_info, args=(sf_context, q, cfg))
    p.start()

    env_info = q.get()
    p.join()

    if cfg.use_env_info_cache:
        with open(cache_filename, "wb") as fobj:
            pickle.dump(env_info, fobj)

    return env_info
