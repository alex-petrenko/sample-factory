from __future__ import annotations

import multiprocessing
import os
import pickle
from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Optional

import gym

from sample_factory.algo.utils.action_distributions import calc_num_actions
from sample_factory.algo.utils.context import set_global_context, sf_global_context
from sample_factory.algo.utils.make_env import BatchedVecEnv, NonBatchedVecEnv, make_env_func_batched
from sample_factory.envs.env_utils import get_default_reward_shaping
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log, project_tmp_dir

ENV_INFO_PROTOCOL_VERSION = 1


@dataclass
class EnvInfo:
    obs_space: gym.Space
    action_space: gym.Space
    num_agents: int
    gpu_actions: bool  # whether actions provided by the agent should be on GPU or not
    gpu_observations: bool  # whether environment provides data (obs, etc.) on GPU or not
    action_splits: List[int]  # in the case of tuple actions, the splits for the actions
    all_discrete: bool  # in the case of tuple actions, whether the actions are all discrete
    frameskip: int
    # potentially customizable reward shaping, a map of reward component names to their respective weights
    # this can be used by PBT to optimize the reward shaping towards a sparse final objective
    reward_shaping_scheme: Optional[Dict[str, float]] = None

    # version of the protocol, used to detect changes in the EnvInfo class and invalidate the cache if needed
    # bump this version if you make any changes to the EnvInfo class
    env_info_protocol_version: Optional[int] = None


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
