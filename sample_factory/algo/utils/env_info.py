import multiprocessing
import os
import pickle
from dataclasses import dataclass
from os.path import join
from typing import List

import gym

from sample_factory.algo.utils.action_distributions import calc_num_actions
from sample_factory.algo.utils.context import set_global_context, sf_global_context
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.utils import AttrDict, experiment_dir, log


@dataclass
class EnvInfo:
    obs_space: gym.Space
    action_space: gym.Space
    num_agents: int
    gpu_actions: bool  # whether actions provided by the agent should be on GPU or not
    action_splits: List[int]  # in the case of tuple actions, the splits for the actions
    all_discrete: bool  # in the case of tuple actions, whether the actions are all discrete
    frameskip: int


def extract_env_info(env, cfg):
    obs_space = env.observation_space
    action_space = env.action_space
    num_agents = env.num_agents

    gpu_actions = cfg.env_gpu_actions

    frameskip = cfg.env_frameskip

    # TODO: PBT stuff (default reward shaping)
    # self.reward_shaping_scheme = None
    # if self.cfg.with_pbt:
    #     self.reward_shaping_scheme = get_default_reward_shaping(tmp_env)

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
        action_splits=action_splits,
        all_discrete=all_discrete,
        frameskip=frameskip,
    )
    return env_info


def spawn_tmp_env_and_get_info(sf_context, res_queue, cfg):
    set_global_context(sf_context)

    tmp_env = make_env_func_batched(cfg, env_config=None)
    env_info = extract_env_info(tmp_env, cfg)  # TODO type errors
    tmp_env.close()
    del tmp_env

    log.debug("Env info: %r", env_info)
    res_queue.put(env_info)


def obtain_env_info_in_a_separate_process(cfg: AttrDict):
    cache_filename = join(experiment_dir(cfg=cfg), f"env_info_{cfg.env}")
    if os.path.isfile(cache_filename):
        with open(cache_filename, "rb") as fobj:
            env_info = pickle.load(fobj)
            return env_info

    sf_context = sf_global_context()

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=spawn_tmp_env_and_get_info, args=(sf_context, q, cfg))
    p.start()

    env_info = q.get()
    p.join()

    with open(cache_filename, "wb") as fobj:
        pickle.dump(env_info, fobj)

    return env_info
