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
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.algorithms.appo.appo_utils import make_env_func
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


def obtain_env_info(cfg: AttrDict):
    gpu_actions = cfg.env_gpu_actions

    # Perhaps run this in a separate process for environments that allocate some exclusive process-wise resources
    # in ctor (such as CUDA contexts)
    # Otherwise this can cause problems, i.e. with envs such as Isaac Gym

    tmp_env = make_env_func(cfg, env_config=None)
    obs_space = tmp_env.observation_space
    action_space = tmp_env.action_space
    num_agents = tmp_env.num_agents
    integer_actions = is_integer_action_env(action_space)

    frameskip = 4 if cfg.env.startswith('doom') else 1  # TODO: this is a hack! rewrite this code!
    log.warning('Assuming frameskip %d! This is a hack. TODO', frameskip)

    # TODO: PBT stuff (default reward shaping)
    # self.reward_shaping_scheme = None
    # if self.cfg.with_pbt:
    #     self.reward_shaping_scheme = get_default_reward_shaping(tmp_env)

    tmp_env.close()

    return EnvInfo(obs_space, action_space, num_agents, gpu_actions, integer_actions, frameskip)


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
def make_model(cfg: AttrDict, env_info: EnvInfo, device: torch.device, timing: Timing) -> _ActorCriticBase:
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

    timing = Timing()

    env_info = obtain_env_info(cfg)

    device = init_device(cfg)

    actor_critic = make_model(cfg, env_info, device, timing)

    buffer_mgr = BufferMgr(cfg, env_info, device)

    comm_broker = SyncCommBroker()

    sampler = SyncSampler(cfg, env_info, comm_broker, actor_critic, device, buffer_mgr)
    sampler.init()

    batcher = SequentialBatcher(buffer_mgr.trajectories_per_batch, buffer_mgr.total_num_trajectories)

    learner = Learner(cfg, env_info, comm_broker, actor_critic, device, buffer_mgr, policy_id=0)  # currently support only single-policy learning
    learner.init()

    runner = SyncRunner(cfg, comm_broker, sampler, batcher, learner)
    status = runner.run()
    return status


def main():
    """RL training entry point."""
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
