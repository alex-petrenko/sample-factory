from __future__ import annotations

import numbers
from queue import Empty
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.sampling.sampling_utils import VectorEnvRunner, record_episode_statistics_wrapper_stats
from sample_factory.algo.utils.env_info import EnvInfo, check_env_info
from sample_factory.algo.utils.make_env import BatchedVecEnv, SequentialVectorizeWrapper, make_env_func_batched
from sample_factory.algo.utils.misc import EPISODIC, POLICY_ID_KEY
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.torch_utils import synchronize
from sample_factory.envs.env_utils import (
    TrainingInfoInterface,
    find_training_info_interface,
    set_reward_shaping,
    set_training_info,
)
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.dicts import iterate_recursively_with_prefix
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import log


def preprocess_actions(env_info: EnvInfo, actions: Tensor | np.ndarray) -> Tensor | np.ndarray | List:
    """
    We expect actions to have shape [num_envs, num_actions].
    For environments that require only one action per step we just squeeze the second dimension,
    because in this case the action is usually expected to be a scalar.

    A potential way to reduce this complexity: demand all environments to have a Tuple action space even if they
    only have a single Discrete or Box action space.
    """

    if env_info.all_discrete or isinstance(env_info.action_space, gym.spaces.Discrete):
        return process_action_space(actions, env_info.gpu_actions, is_discrete=True)
    elif isinstance(env_info.action_space, gym.spaces.Box):
        return process_action_space(actions, env_info.gpu_actions, is_discrete=False)
    elif isinstance(env_info.action_space, gym.spaces.Tuple):
        # input is (num_envs, num_actions)
        out_actions = []
        for split, space in zip(torch.split(actions, env_info.action_splits, 1), env_info.action_space):
            out_actions.append(
                process_action_space(split, env_info.gpu_actions, isinstance(space, gym.spaces.Discrete))
            )
        # this line can be used to transpose the actions, perhaps add as an option ?
        # out_actions = list(zip(*out_actions)) # transpose
        return out_actions

    raise NotImplementedError(f"Unknown action space type: {env_info.action_space}")


def process_action_space(actions: torch.Tensor, gpu_actions: bool, is_discrete: bool):
    if is_discrete:
        actions = actions.to(torch.int32)
    if not gpu_actions:
        actions = actions.cpu().numpy()

    # action tensor/array should have two dimensions (num_agents, num_actions) where num_agents is a number of
    # individual actors in a vectorized environment (whether actually different agents or separate envs - does not
    # matter)
    # While continuous action envs generally expect an array/tensor of actions, even when there's just one action,
    # discrete action envs typically expect to get the action index when there's only one action. So we squeeze the
    # second dimension for integer action envs.
    assert actions.ndim == 2, f"Expected actions to have two dimensions, got {actions}"
    if is_discrete and actions.shape[1] == 1:
        actions = actions.squeeze(-1)

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
        self,
        cfg,
        env_info,
        num_envs,
        worker_idx,
        split_idx,
        buffer_mgr,
        sampling_device: str,
        training_info: List[Optional[Dict]],
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
        :param training_info: curr env steps, reward shaping scheme, etc.
        """
        super().__init__(cfg, env_info, worker_idx, split_idx, buffer_mgr, sampling_device)

        self.policy_id = worker_idx % self.cfg.num_policies
        log.debug(f"EnvRunner {worker_idx}-{split_idx} uses policy {self.policy_id}")

        self.num_envs = num_envs

        self.vec_env: Optional[BatchedVecEnv | SequentialVectorizeWrapper] = None
        self.env_training_info_interface: Optional[TrainingInfoInterface] = None

        self.last_obs = None
        self.last_rnn_state = None
        self.policy_id_buffer = None

        self.curr_traj: Optional[TensorDict] = None
        self.curr_step: Optional[TensorDict] = None
        self.curr_traj_slice: Optional[slice] = None

        self.curr_episode_reward = self.curr_episode_len = None

        self.training_info: List[Optional[Dict]] = training_info

        self.min_raw_rewards = self.max_raw_rewards = None

        self.device: Optional[torch.device] = None

    def init(self, timing):
        """
        Actually instantiate the env instances.
        Also creates ActorState objects that hold the state of individual actors in (potentially) multi-agent envs.
        """
        envs: List[BatchedVecEnv] = []
        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i

            # global env id within the entire system
            env_id = self.worker_idx * self.cfg.num_envs_per_worker + vector_idx

            env_config = AttrDict(
                worker_index=self.worker_idx,
                vector_index=vector_idx,
                env_id=env_id,
            )

            # log.info('Creating env %r... %d-%d-%d', env_config, self.worker_idx, self.split_idx, env_i)
            # a vectorized environment - we assume that it always provides a dict of vectors of obs, rewards, etc.
            env: BatchedVecEnv = make_env_func_batched(self.cfg, env_config)
            check_env_info(env, self.env_info, self.cfg)

            env.seed(env_id)  # since Gym 0.26 seeding is done in reset(), we do it in BatchedVecEnv class
            envs.append(env)

        if len(envs) == 1:
            # assuming this is already a vectorized environment
            assert envs[0].num_agents >= 1  # sanity check
            self.vec_env = envs[0]
        else:
            self.vec_env = SequentialVectorizeWrapper(envs)

        self.env_training_info_interface = find_training_info_interface(self.vec_env)

        self.last_obs, info = self.vec_env.reset()  # anything we need to do with info? Currently we ignore it

        self.last_rnn_state = torch.zeros_like(self.traj_tensors["rnn_states"][0 : self.vec_env.num_agents, 0])

        # we assume that all data will be on the same device
        self.device = self.last_rnn_state.device

        self.policy_id_buffer = torch.empty_like(self.traj_tensors["policy_id"][0 : self.vec_env.num_agents, 0])
        self.policy_id_buffer[:] = self.policy_id

        assert self.rollout_step == 0

        self.curr_episode_reward = torch.zeros(self.vec_env.num_agents)
        self.curr_episode_len = torch.zeros(self.vec_env.num_agents, dtype=torch.int32)
        self.min_raw_rewards = torch.empty_like(self.curr_episode_reward).fill_(np.inf)
        self.max_raw_rewards = torch.empty_like(self.curr_episode_reward).fill_(-np.inf)

        self.env_step_ready = True

    def _process_rewards(self, rewards_orig: Tensor, rewards_orig_cpu: Tensor) -> Tensor:
        rewards = rewards_orig * self.cfg.reward_scale
        rewards.clamp_(-self.cfg.reward_clip, self.cfg.reward_clip)
        self.min_raw_rewards = torch.min(self.min_raw_rewards, rewards_orig_cpu)
        self.max_raw_rewards = torch.max(self.max_raw_rewards, rewards_orig_cpu)
        return rewards

    def _process_env_step(self, rewards: Tensor, dones_orig: Tensor, infos):
        dones = dones_orig.cpu()
        num_dones = dones.sum().item()

        self.curr_episode_reward += rewards
        self.curr_episode_len += self.env_info.frameskip if self.cfg.summaries_use_frameskip else 1

        reports = []
        if num_dones <= 0:
            return reports

        finished = dones.nonzero(as_tuple=True)[0]

        stats = dict(
            reward=self.curr_episode_reward[finished],
            len=self.curr_episode_len[finished],
            min_raw_reward=self.min_raw_rewards[finished],
            max_raw_reward=self.max_raw_rewards[finished],
        )

        if isinstance(infos, dict):
            # vectorized reports
            for _, key, value, prefix in iterate_recursively_with_prefix(infos):
                key_str = key
                if prefix:
                    key_str = f"{'/'.join(prefix)}/{key}"

                if isinstance(value, Tensor):
                    if value.numel() == 1:
                        stats[key_str] = value.item()
                    elif len(value.shape) >= 1 and len(value) == self.vec_env.num_agents:
                        # saving value for all agents who finished the episode
                        stats[key_str] = value[finished]
                    else:
                        log.warning(f"Infos tensor with unexpected shape {value.shape}")
                elif isinstance(value, numbers.Number):
                    stats[key_str] = value
        else:
            # non-vectorized reports: TODO (parse infos)

            assert isinstance(infos, (list, tuple)), "Expect infos to be a list or tuple of dicts"

            # some envs like Atari use a special wrapper to record episode statistics
            stats_rew, stats_len = [], []
            for agent_i in finished.tolist():
                episode_wrapper_stats = record_episode_statistics_wrapper_stats(infos[agent_i])
                if episode_wrapper_stats is not None:
                    wrapper_rew, wrapper_len = episode_wrapper_stats
                    stats_rew.append(wrapper_rew)
                    stats_len.append(wrapper_len)

            if stats_rew and stats_len:
                # length of these might not match len(finished), but stats handler does not care
                stats["RecordEpisodeStatistics_reward"] = np.array(stats_rew)
                stats["RecordEpisodeStatistics_len"] = np.array(stats_len)

        # make sure everything in the dict is either a scalar or a numpy array
        for key, value in stats.items():
            if isinstance(value, Tensor):
                stats[key] = value.cpu().numpy()
            else:
                assert isinstance(value, np.ndarray) or isinstance(
                    value, numbers.Number
                ), f"Expect stats[{key}] to be a scalar or numpy array, got {type(value)}"

        reports.append({EPISODIC: stats, POLICY_ID_KEY: self.policy_id})

        self.curr_episode_reward[finished] = 0
        self.curr_episode_len[finished] = 0
        self.min_raw_rewards[finished] = np.inf
        self.max_raw_rewards[finished] = -np.inf

        return reports

    def _finalize_trajectories(self) -> List[Dict]:
        # Saving obs and hidden states for the step AFTER the last step in the current rollout.
        # We're going to need them later when we calculate next step value estimates.
        self.curr_traj["obs"][:, self.cfg.rollout] = self.last_obs
        self.curr_traj["rnn_states"][:, self.cfg.rollout] = self.last_rnn_state

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
        with timing.add_time("process_policy_outputs"):
            # save actions/logits/values etc. for the current rollout step
            self.curr_step[:] = self.policy_output_tensors
            actions = preprocess_actions(self.env_info, self.policy_output_tensors["actions"])

        complete_rollouts, episodic_stats = [], []

        with timing.add_time("env_step"):
            self.last_obs, rewards, terminated, truncated, infos = self.vec_env.step(actions)
            dones = terminated | truncated  # both should be either tensors or numpy arrays of bools

        with timing.add_time("post_env_step"):
            self.policy_id_buffer[:] = self.policy_id

            # record the results from the env step
            rewards_cpu = rewards.cpu()
            processed_rewards = self._process_rewards(rewards, rewards_cpu)
            self.curr_step[:] = dict(
                rewards=processed_rewards,
                dones=dones,
                time_outs=truncated,  # true only when done is also true, used for value bootstrapping
                policy_id=self.policy_id_buffer,
            )

            # reset next-step hidden states to zero if we encountered an episode boundary
            # not sure if this is the best practice, but this is what everybody seems to be doing
            not_done = (1.0 - self.curr_step["dones"].float()).unsqueeze(-1)
            self.last_rnn_state = self.policy_output_tensors["new_rnn_states"] * not_done

            with timing.add_time("process_env_step"):
                stats = self._process_env_step(rewards_cpu, dones, infos)
            episodic_stats.extend(stats)

        self.rollout_step += 1

        with timing.add_time("finalize_trajectories"):
            if self.rollout_step == self.cfg.rollout:
                # finalize and serialize the trajectory if we have a complete rollout
                complete_rollouts = self._finalize_trajectories()
                self.rollout_step = 0
                # we will need to request a new trajectory buffer!
                self.curr_traj_slice = self.curr_traj = None

                if self.training_info[self.policy_id] is not None:
                    reward_shaping = self.training_info[self.policy_id].get("reward_shaping", None)
                    set_reward_shaping(self.vec_env, reward_shaping, slice(0, self.vec_env.num_agents))
                    set_training_info(self.env_training_info_interface, self.training_info[self.policy_id])

        self.env_step_ready = True
        return complete_rollouts, episodic_stats

    def update_trajectory_buffers(self, timing) -> bool:
        if self.curr_traj_slice is not None and self.curr_traj is not None:
            # don't need to do anything - we have a trajectory buffer already
            return True

        with timing.add_time("wait_for_trajectories"):
            try:
                buffers = self.traj_buffer_queue.get(block=False, timeout=1e9)
            except Empty:
                return False

            self.curr_traj_slice = buffers
            self.curr_traj = self.traj_tensors[self.curr_traj_slice]
            return True

    def generate_policy_request(self) -> Optional[Dict]:
        if not self.env_step_ready:
            # we haven't actually simulated the environment yet
            return None

        if self.curr_traj is None:
            # we don't have a shared buffer to store data in - still waiting for one to become available
            return None

        self.curr_step = self.curr_traj[:, self.rollout_step]
        # save observations and RNN states in a trajectory
        self.curr_step[:] = dict(obs=self.last_obs, rnn_states=self.last_rnn_state)
        policy_request = {self.policy_id: (self.curr_traj_slice, self.rollout_step)}
        self.env_step_ready = False
        return policy_request

    def synchronize_devices(self) -> None:
        """Make sure all writes to shared device buffers are finished."""
        synchronize(self.cfg, self.device)

    def close(self):
        self.vec_env.close()
