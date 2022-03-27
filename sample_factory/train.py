import multiprocessing
import sys
from dataclasses import dataclass

import gym
import torch
from gym.spaces import Discrete, Tuple

from sample_factory.algo.batchers.batcher_sequential import SequentialBatcher
from sample_factory.algo.learners.learner import Learner
from sample_factory.algo.runners.runner_sync import SyncRunner
from sample_factory.algo.samplers.sampler_sync import SyncSampler
from sample_factory.algo.utils.communication_broker import SyncCommBroker
from sample_factory.algo.utils.context import sf_global_context, set_global_context
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.algorithms.appo.appo_utils import make_env_func, make_env_func_v2
from sample_factory.algorithms.appo.model import _ActorCriticBase, create_actor_critic
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.algorithms.utils.spaces.discretized import Discretized
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import AttrDict, log


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
    sf_context = sf_global_context()

    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=spawn_tmp_env_and_get_info, args=(sf_context, q, cfg))
    p.start()

    env_info = q.get()
    p.join()

    return env_info


# TODO: all components should use the same function
def init_device(cfg: AttrDict) -> torch.device:
    if cfg.device == 'gpu':
        torch.backends.cudnn.benchmark = True  # TODO: this probably shouldn't be here

        # we should already see only one CUDA device, because of env vars
        assert torch.cuda.device_count() == 1
        device = torch.device('cuda', index=0)  # TODO: other devices? add to cfg?
    else:
        device = torch.device('cpu')

    return device


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
    # TODO: temporarily use stubs to figure out the interfaces
    # once everything stub components can communicate, we're ready to start development of actual components

    env_info = obtain_env_info_in_a_separate_process(cfg)
    device = init_device(cfg)
    actor_critic = make_model(cfg, env_info, device)
    buffer_mgr = BufferMgr(cfg, env_info, device)

    runner = SyncRunner(cfg)
    evt_loop = runner.event_loop
    # evt_loop.verbose = True

    sampler = SyncSampler(evt_loop, cfg, env_info, actor_critic, device, buffer_mgr, runner.timing)
    sampler.init()

    batcher = SequentialBatcher(evt_loop, buffer_mgr.trajectories_per_batch, buffer_mgr.total_num_trajectories)

    learner = Learner(evt_loop, cfg, env_info, actor_critic, device, buffer_mgr, policy_id=0, timing=runner.timing)  # currently support only single-policy learning
    learner.init()

    runner.init(sampler, batcher, learner)
    status = runner.run()
    return status


def main():
    """RL training entry point."""
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
