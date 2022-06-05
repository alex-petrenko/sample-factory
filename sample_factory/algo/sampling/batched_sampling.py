from __future__ import annotations

from queue import Empty
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
from sample_factory.algorithms.utils.pytorch_utils import to_scalar
from torch import Tensor

from sample_factory.algo.sampling.sampling_utils import VectorEnvRunner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.tensor_utils import clone_tensor
from sample_factory.algorithms.appo.appo_utils import SequentialVectorizeWrapper, make_env_func_batched
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import AttrDict, log


# TODO: remove code duplication (actor_worker.py)
def preprocess_actions(env_info: EnvInfo, actions: Tensor | np.ndarray):
    if env_info.integer_actions:
            actions = actions.to(torch.int32)  # is it faster to do on GPU or CPU?

    if not env_info.gpu_actions:
        actions = actions.cpu().numpy()

    # TODO: do we need this? actions are a tensor of size [batch_size, action_shape] (or just [batch_size] if it is a single action per env)
    # if len(actions) == 1:
    #     actions = actions.item()

    return actions


class BatchedVectorEnvRunner(VectorEnvRunner):
    # TODO: comment
    """
    A collection of environments simulated sequentially.
    With double buffering each actor worker holds two vector runners and switches between them.
    Without single buffering we only use a single VectorEnvRunner per actor worker.

    All envs on a single VectorEnvRunner run in unison, e.g. they all do one step at a time together.
    This also means they all finish their rollouts together. This allows us to minimize the amount of messages
    passed around.

    Individual envs (or agents in these envs in case of multi-agent) can potentially be controlled by different
    policies when we're doing PBT. We only start simulating the next step in the environment when
    all actions from all envs and all policies are collected. This leaves optimization potential: we can start
    simulating some envs right away as actions for them arrive. But usually double-buffered sampling masks
    this type of inefficiency anyway. The worker is probably still rendering a previous vector of envs when
    the actions arrive.
    """

    def __init__(
            self, cfg, env_info, num_envs, worker_idx, split_idx,
            buffer_mgr, sampling_device: str, pbt_reward_shaping,  #TODO pbt reward
    ):
        # TODO: comment
        """
        Ctor.

        :param cfg: global system config (all CLI params)
        :param num_envs: number of envs to run in this vector runner
        :param worker_idx: idx of the parent worker
        :param split_idx: index of the environment group in double-buffered sampling (either 0 or 1). Always 0 when
        double-buffered sampling is disabled.
        :param buffer_mgr: a collection of all shared data structures used by the algorithm. Most importantly,
        the trajectory buffers in shared memory.
        :param pbt_reward_shaping: initial reward shaping dictionary, for configuration where PBT optimizes
        reward coefficients in environments.
        """
        super().__init__(cfg, env_info, worker_idx, split_idx, buffer_mgr, sampling_device)

        self.policy_id = worker_idx % self.cfg.num_policies
        log.debug(f'EnvRunner {worker_idx}-{split_idx} uses policy {self.policy_id}')

        self.num_envs = num_envs

        self.vec_env = None
        self.last_obs = None
        self.last_rnn_state = None
        self.policy_id_buffer = None

        self.curr_traj: Optional[TensorDict] = None
        self.curr_step: Optional[TensorDict] = None
        self.curr_traj_slice: Optional[slice] = None

        self.curr_episode_reward = self.curr_episode_len = None

        self.pbt_reward_shaping = pbt_reward_shaping  # TODO

        self.min_raw_rewards = self.max_raw_rewards = None
        self.num_timeouts = self.num_dones = 0

    def init(self, timing) -> Dict:
        """
        Actually instantiate the env instances.
        Also creates ActorState objects that hold the state of individual actors in (potentially) multi-agent envs.
        """
        envs = []
        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i

            # global env id within the entire system
            env_id = self.worker_idx * self.cfg.num_envs_per_worker + vector_idx

            env_config = AttrDict(
                worker_index=self.worker_idx, vector_index=vector_idx, env_id=env_id,
            )

            # log.info('Creating env %r... %d-%d-%d', env_config, self.worker_idx, self.split_idx, env_i)
            # a vectorized environment - we assume that it always provides a dict of vectors of obs, rewards, dones, infos
            env = make_env_func_batched(self.cfg, env_config=env_config)

            env.seed(env_id)
            envs.append(env)

        if len(envs) == 1:
            self.vec_env = envs[0]
        else:
            self.vec_env = SequentialVectorizeWrapper(envs)

        self.update_trajectory_buffers(timing)
        assert self.curr_traj is not None and self.curr_traj_slice is not None

        self.last_obs = self.vec_env.reset()
        self.last_rnn_state = clone_tensor(self.traj_tensors['rnn_states'][0:self.vec_env.num_agents, 0])
        self.last_rnn_state[:] = 0.0

        self.policy_id_buffer = clone_tensor(self.traj_tensors['policy_id'][0:self.vec_env.num_agents, 0])
        self.policy_id_buffer[:] = self.policy_id

        assert self.rollout_step == 0

        self.curr_episode_reward = torch.zeros(self.vec_env.num_agents)
        self.curr_episode_len = torch.zeros(self.vec_env.num_agents, dtype=torch.int32)
        self.min_raw_rewards = torch.empty_like(self.curr_episode_reward).fill_(np.inf)
        self.max_raw_rewards = torch.empty_like(self.curr_episode_reward).fill_(-np.inf)

        self.env_step_ready = True
        policy_request = self.generate_policy_request(timing)
        assert policy_request is not None
        return policy_request

    def _process_rewards(self, rewards_orig: Tensor, rewards_orig_cpu: Tensor, infos: Dict[Any, Any], values: Tensor, dones: Tensor):
        rewards = rewards_orig * self.cfg.reward_scale
        rewards.clamp_(-self.cfg.reward_clip, self.cfg.reward_clip)
        self.min_raw_rewards = torch.min(self.min_raw_rewards, rewards_orig_cpu)
        self.max_raw_rewards = torch.max(self.max_raw_rewards, rewards_orig_cpu)

        time_outs = infos.get('time_outs')
        if time_outs is not None:
            if self.cfg.value_bootstrap:
                # What we really want here is v(t+1) which we don't have, using v(t) is an approximation that
                # requires that rew(t) can be generally ignored.
                # TODO: if gamma is modified by PBT it should be updated here too?!
                rewards.add_(self.cfg.gamma * values * time_outs.float())

            time_outs = time_outs.long()
            num_timeouts = time_outs.sum().item()
            dones = dones.long()
            num_dones = dones.sum().item()
            self.num_timeouts += num_timeouts
            self.num_dones += num_dones

            assert (time_outs & dones).sum().item() == num_timeouts, \
                f'Every timeout should correspond to a done flag {time_outs.nonzero().squeeze()=} vs {dones.nonzero().squeeze()=}'
            assert num_timeouts <= num_dones, f'Timeouts should correspond to dones {num_timeouts=} vs {num_dones=}'

        return rewards

    def _process_env_step(self, rewards: Tensor, dones_orig: Tensor, infos):
        dones = dones_orig.cpu()
        num_dones = dones.sum().item()

        self.curr_episode_reward += rewards
        self.curr_episode_len += 1

        reports = []
        if num_dones > 0:
            finished = dones.nonzero(as_tuple=True)[0]

            # TODO: get rid of the loop (we can do it vectorized)
            # TODO: remove code duplication

            last_rew = self.curr_episode_reward[finished]
            last_dur = self.curr_episode_len[finished]
            last_min_rew = self.min_raw_rewards[finished]
            last_max_rew = self.max_raw_rewards[finished]

            for i in range(num_dones):
                last_episode_true_objective = last_rew[i]
                last_episode_extra_stats = None

                # TODO: we somehow need to deal with two cases: when infos is a dict of tensors and when it is a list of dicts
                # this only handles the latter.
                if isinstance(infos, (list, tuple)):
                    last_episode_true_objective = infos[finished[i]].get('true_objective', last_rew[i])
                    last_episode_extra_stats = infos[finished[i]].get('episode_extra_stats', None)

                stats = dict(
                    reward=last_rew[i],
                    len=last_dur[i],
                    true_objective=last_episode_true_objective,
                    min_raw_reward=last_min_rew[i],
                    max_raw_reward=last_max_rew[i],
                )
                if last_episode_extra_stats:
                    stats['episode_extra_stats'] = last_episode_extra_stats

                for k, v in stats.items():
                    stats[k] = to_scalar(v)

                report = dict(episodic=stats, policy_id=self.policy_id)
                reports.append(report)

            self.curr_episode_reward[finished] = 0
            self.curr_episode_len[finished] = 0
            self.min_raw_rewards[finished] = np.inf
            self.max_raw_rewards[finished] = -np.inf

        return reports

    def _finalize_trajectories(self) -> List[Dict]:
        # Saving obs and hidden states for the step AFTER the last step in the current rollout.
        # We're going to need them later when we calculate next step value estimates.
        self.curr_traj['obs'][:, self.cfg.rollout] = self.last_obs
        self.curr_traj['rnn_states'][:, self.cfg.rollout] = self.last_rnn_state

        traj_dict = dict(policy_id=self.policy_id, traj_buffer_idx=self.curr_traj_slice)
        return [traj_dict]

    def advance_rollouts(self, policy_id: PolicyID, timing) -> Tuple[List[Dict], List[Dict]]:
        # TODO: comment
        """
        Main function in VectorEnvRunner. Does one step of simulation (if all actions for all actors are available).

        :param policy_id:
        :param timing: this is just for profiling
        :return: same as reset(), return a set of requests for policy workers, asking them to generate actions for
        the next env step.
        """
        with timing.add_time('process_policy_outputs'):
            # save actions/logits/values etc. for the current rollout step
            self.curr_step[:] = self.policy_output_tensors  # TODO: output tensors should contain the policy version
            actions = preprocess_actions(self.env_info, self.policy_output_tensors['actions'])

        complete_rollouts, episodic_stats = [], []  # TODO

        with timing.add_time('env_step'):
            self.last_obs, rewards, dones, infos = self.vec_env.step(actions)

        with timing.add_time('post_env_step'):
            self.policy_id_buffer[:] = self.policy_id

            # TODO: for vectorized envs we either have a dictionary of tensors (isaacgym_examples), or a list of dictionaries (i.e. swarm_rl quadrotors)
            # Need an adapter class so it's consistent, i.e. always a dict of tensors.
            # this should yield indices of inactive agents
            #
            # if infos:
            #     inactive_agents = [i for i, info in enumerate(infos) if not info.get('is_active', True)]
            #     self.policy_id_buffer[inactive_agents] = -1
            # TODO: batcher runner probably won't have inactive agent support for now.

            # record the results from the env step
            rewards_cpu = rewards.cpu()
            processed_rewards = self._process_rewards(rewards, rewards_cpu, infos, self.policy_output_tensors['values'], dones)

            self.curr_step[:] = dict(rewards=processed_rewards, dones=dones, policy_id=self.policy_id_buffer)

            # reset next-step hidden states to zero if we encountered an episode boundary
            # not sure if this is the best practice, but this is what everybody seems to be doing
            not_done = (1.0 - self.curr_step['dones'].float()).unsqueeze(-1)
            self.last_rnn_state = self.policy_output_tensors['new_rnn_states'] * not_done

            with timing.add_time('process_env_step'):
                stats = self._process_env_step(rewards_cpu, dones, infos)
            episodic_stats.extend(stats)

        self.rollout_step += 1

        with timing.add_time('finalize_trajectories'):
            if self.rollout_step == self.cfg.rollout:
                # finalize and serialize the trajectory if we have a complete rollout
                complete_rollouts = self._finalize_trajectories()
                self.rollout_step = 0
                # we will need to request a new trajectory buffer!
                self.curr_traj_slice = self.curr_traj = None

                if self.num_dones > 0:
                    timeout_fraction = self.num_timeouts / self.num_dones
                    # this will be processed by the regular episodic stats handler
                    episodic_stats.append(dict(episodic=dict(timeout_fraction=timeout_fraction), policy_id=self.policy_id))

                # these stats will be per rollout
                self.num_dones = self.num_timeouts = 0

        self.env_step_ready = True
        return complete_rollouts, episodic_stats

    def update_trajectory_buffers(self, timing, block=False) -> bool:
        if self.curr_traj_slice is not None and self.curr_traj is not None:
            # don't need to do anything - we have a trajectory buffer already
            return True

        with timing.add_time('wait_for_trajectories'):
            try:
                buffers = self.traj_buffer_queue.get(block=block, timeout=1e9)
            except Empty:
                return False

            self.curr_traj_slice = buffers
            self.curr_traj = self.traj_tensors[self.curr_traj_slice]
            return True

    def generate_policy_request(self, timing) -> Optional[Dict]:
        if not self.env_step_ready:
            # we haven't actually simulated the environment yet
            return None

        if self.curr_traj is None:
            # we don't have a shared buffer to store data in - still waiting for one to become available
            return None

        with timing.add_time('prepare_next_step'):
            self.curr_step = self.curr_traj[:, self.rollout_step]
            # save observations and RNN states in a trajectory
            self.curr_step[:] = dict(obs=self.last_obs, rnn_states=self.last_rnn_state)
            policy_request = {self.policy_id: (self.curr_traj_slice, self.rollout_step)}
            self.env_step_ready = False

        return policy_request

    def close(self):
        self.vec_env.close()
