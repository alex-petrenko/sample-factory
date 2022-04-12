import multiprocessing
import os
import pickle
import sys
from dataclasses import dataclass
from os.path import join

import gym
import torch
from gym.spaces import Discrete, Tuple

from sample_factory.algo.batchers.batcher_sequential import SequentialBatcher
from sample_factory.algo.learners.learner import Learner, init_learner_process
from sample_factory.algo.runners.runner_sync import SyncRunner
from sample_factory.algo.samplers.sampler_sync import SyncSampler, init_sampler_process
from sample_factory.algo.utils.context import sf_global_context, set_global_context
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.algo.utils.torch_utils import init_torch_runtime
from sample_factory.algorithms.appo.appo_utils import make_env_func_v2, set_global_cuda_envvars
from sample_factory.algorithms.appo.model import _ActorCriticBase, create_actor_critic
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.algorithms.utils.spaces.discretized import Discretized
from sample_factory.signal_slot.signal_slot import EventLoopProcess
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import AttrDict, log, experiment_dir


# TODO: remove the other version in actor_worker.py
def is_integer_action_env(action_space):
    integer_actions = False
    if isinstance(action_space, (Discrete, Discretized)):
        integer_actions = True
    if isinstance(action_space, Tuple):
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

    tmp_env = make_env_func_v2(cfg, env_config=None)
    env_info = extract_env_info(tmp_env, cfg)
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


# TODO: refactor this - all algorithms should use the same function
def make_model(cfg: AttrDict, env_info: EnvInfo, device: torch.device, timing: Timing = None) -> _ActorCriticBase:
    actor_critic = create_actor_critic(cfg, env_info.obs_space, env_info.action_space, timing)
    actor_critic.model_to_device(device)
    # self.actor_critic.share_memory()

    # TODO: implement this stuff later
    # if cfg.use_cpc:
    #     aux_loss_module = CPCA(cfg, env_info.action_space)
    #
    # if self.aux_loss_module is not None:
    #     self.aux_loss_module.to(device=self.device)

    return actor_critic


def run_rl(cfg):
    if cfg.serial_mode:
        return run_rl_serial(cfg)
    else:
        return run_rl_async(cfg)


def run_rl_serial(cfg):
    set_global_cuda_envvars(cfg)
    init_torch_runtime(cfg)  # in serial mode everything will be happening in the main process, so we need to initialize cuda

    env_info = obtain_env_info_in_a_separate_process(cfg)

    buffer_mgr = BufferMgr(cfg, env_info)

    runner = SyncRunner(cfg)
    # evt_loop.verbose = True

    learner = Learner(runner.event_loop, cfg, env_info, buffer_mgr, policy_id=0)  # currently support only single-policy learning
    batcher = SequentialBatcher(runner.event_loop, buffer_mgr.trajectories_per_batch, buffer_mgr.total_num_trajectories, env_info)
    sampler = SyncSampler(runner.event_loop, cfg, env_info, learner.param_server, buffer_mgr, batcher.sampling_batches_queue)

    runner.init(sampler, batcher, learner)
    status = runner.run()
    return status


# TODO: remove duplicate code
def run_rl_async(cfg):
    set_global_cuda_envvars(cfg)

    env_info = obtain_env_info_in_a_separate_process(cfg)

    buffer_mgr = BufferMgr(cfg, env_info)

    runner = SyncRunner(cfg)

    policy_id = 0  # TODO: multiple policies
    ctx = multiprocessing.get_context('spawn')
    learner_proc = EventLoopProcess('learner_proc', ctx, init_func=init_learner_process, args=(sf_global_context(), cfg, policy_id))
    batcher = SequentialBatcher(learner_proc.event_loop, buffer_mgr.trajectories_per_batch, buffer_mgr.total_num_trajectories, env_info)
    learner = Learner(learner_proc.event_loop, cfg, env_info, buffer_mgr, policy_id=0, mp_ctx=ctx)  # currently support only single-policy learning

    sampler_proc = EventLoopProcess('sampler_proc', ctx, init_func=init_sampler_process, args=(sf_global_context(), cfg, policy_id))
    sampler = SyncSampler(sampler_proc.event_loop, cfg, env_info, learner.param_server, buffer_mgr, batcher.sampling_batches_queue)

    runner.init(sampler, batcher, learner)

    sampler_proc.start()
    learner_proc.start()
    status = runner.run()
    learner_proc.join()
    sampler_proc.join()

    return status


def main():
    """RL training entry point."""
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
