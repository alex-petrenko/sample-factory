# TODO: remove the other version in actor_worker.py
import multiprocessing
import os
import pickle
from dataclasses import dataclass
from os.path import join

import gym
from gym.spaces import Discrete

from sample_factory.algo.utils.context import set_global_context, sf_global_context
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.spaces.discretized import Discretized
from sample_factory.utils.utils import log, AttrDict, experiment_dir


def is_integer_action_env(action_space):
    integer_actions = False
    if isinstance(action_space, (Discrete, Discretized)):
        integer_actions = True
    if isinstance(action_space, gym.spaces.Tuple):
        all_subspaces_discrete = all(isinstance(s, (Discrete, Discretized)) for s in action_space.spaces)
        if all_subspaces_discrete:
            integer_actions = True
        else:
            # tecnhically possible to add support for such spaces, but it's untested
            # for now, look at Discretized instead.
            raise Exception(
                'Mixed discrete & continuous action spaces are not supported (should be an easy fix)'
            )

    return integer_actions


@dataclass
class EnvInfo:
    obs_space: gym.Space
    action_space: gym.Space
    num_agents: int
    gpu_actions: bool  # whether actions provided by the agent should be on GPU or not
    integer_actions: bool  # whether actions returned by the policy should be cast to int32 (i.e. for discrete action envs)
    frameskip: int


def extract_env_info(env, cfg):
    obs_space = env.observation_space
    action_space = env.action_space
    num_agents = env.num_agents
    integer_actions = is_integer_action_env(action_space)
    gpu_actions = cfg.env_gpu_actions

    frameskip = 4 if cfg.env.startswith('doom') else 1  # TODO: this is a hack! rewrite this code!
    log.warning('Assuming frameskip %d! This is a hack. TODO', frameskip)

    # TODO: PBT stuff (default reward shaping)
    # self.reward_shaping_scheme = None
    # if self.cfg.with_pbt:
    #     self.reward_shaping_scheme = get_default_reward_shaping(tmp_env)

    env_info = EnvInfo(obs_space, action_space, num_agents, gpu_actions, integer_actions, frameskip)
    return env_info


def spawn_tmp_env_and_get_info(sf_context, res_queue, cfg):
    set_global_context(sf_context)

    tmp_env = make_env_func_batched(cfg, env_config=None)
    env_info = extract_env_info(tmp_env, cfg)  # TODO type errors
    tmp_env.close()
    del tmp_env

    log.debug('Env info: %r', env_info)
    res_queue.put(env_info)


def obtain_env_info_in_a_separate_process(cfg: AttrDict):
    cache_filename = join(experiment_dir(cfg=cfg), f'env_info_{cfg.env}')
    if os.path.isfile(cache_filename):
        with open(cache_filename, 'rb') as fobj:
            env_info = pickle.load(fobj)
            return env_info

    sf_context = sf_global_context()

    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=spawn_tmp_env_and_get_info, args=(sf_context, q, cfg))
    p.start()

    env_info = q.get()
    p.join()

    with open(cache_filename, 'wb') as fobj:
        pickle.dump(env_info, fobj)

    return env_info
