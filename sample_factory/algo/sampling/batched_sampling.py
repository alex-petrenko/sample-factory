from queue import Empty
from typing import Optional, Dict, List, Tuple, Any

import torch
from torch import Tensor

from sample_factory.algo.sampling.sampling_utils import VectorEnvRunner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.tensor_dict import TensorDict, clone_tensor
from sample_factory.algorithms.appo.appo_utils import SequentialVectorizeWrapper, make_env_func_batched
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import AttrDict, log


# TODO: remove code duplication (actor_worker.py)
def preprocess_actions(env_info: EnvInfo, actions: Tensor):
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

        self.env_step_ready = True
        policy_request = self.generate_policy_request(timing)
        assert policy_request is not None
        return policy_request

    def _process_rewards(self, rewards_orig: Tensor, infos: Dict[Any, Any], values: Tensor):
        rewards = rewards_orig * self.cfg.reward_scale
        rewards.clamp_(-self.cfg.reward_clip, self.cfg.reward_clip)

        if self.cfg.value_bootstrap and 'time_outs' in infos:
            # What we really want here is v(t+1) which we don't have, using v(t) is an approximation that
            # requires that rew(t) can be generally ignored.
            # TODO: if gamma is modified by PBT it should be updated here too?!
            rewards.add_(self.cfg.gamma * values * infos['time_outs'].float())

        return rewards

    def _process_env_step(self, rewards_orig, dones_orig, infos):
        rewards = rewards_orig.cpu()
        dones = dones_orig.cpu()

        self.curr_episode_reward += rewards
        self.curr_episode_len += 1

        finished_episodes = dones.nonzero(as_tuple=True)[0]

        # TODO: get rid of the loop (we can do it vectorized)
        # TODO: remove code duplication
        reports = []
        for i in finished_episodes:
            agent_i = i.item()

            last_episode_reward = self.curr_episode_reward[agent_i].item()
            last_episode_duration = self.curr_episode_len[agent_i].item()

            last_episode_true_objective = last_episode_reward
            last_episode_extra_stats = None

            # TODO: we somehow need to deal with two cases: when infos is a dict of tensors and when it is a list of dicts
            # this only handles the latter.
            if isinstance(infos, (list, tuple)):
                last_episode_true_objective = infos[agent_i].get('true_objective', last_episode_reward)
                last_episode_extra_stats = infos[agent_i].get('episode_extra_stats', None)

            stats = dict(reward=last_episode_reward, len=last_episode_duration, true_objective=last_episode_true_objective)
            if last_episode_extra_stats:
                stats['episode_extra_stats'] = last_episode_extra_stats

            report = dict(episodic=stats, policy_id=self.policy_id)
            reports.append(report)

        self.curr_episode_reward[finished_episodes] = 0
        self.curr_episode_len[finished_episodes] = 0
        return reports

    def _finalize_trajectories(self) -> List[Dict]:
        # Saving obs and hidden states for the step AFTER the last step in the current rollout.
        # We're going to need them later when we calculate next step value estimates.
        self.curr_traj['obs'][:, self.cfg.rollout] = self.last_obs
        self.curr_traj['rnn_states'][:, self.cfg.rollout] = self.last_rnn_state

        # TODO: make this consistent with the CPU sampler
        traj_dict = dict(policy_id=self.policy_id, traj_buffer_idx=self.curr_traj_slice)
        return [traj_dict]

    def advance_rollouts(self, policy_id: PolicyID, timing) -> Tuple[List[Dict], List[Dict]]:
        # TODO: comment
        """
        Main function in VectorEnvRunner. Does one step of simulation (if all actions for all actors are available).

        :param timing: this is just for profiling
        :return: same as reset(), return a set of requests for policy workers, asking them to generate actions for
        the next env step.
        """
        assert policy_id == self.policy_id  # TODO: remove

        with timing.add_time('process_policy_outputs'):
            # save actions/logits/values etc. for the current rollout step
            self.curr_step[:] = self.policy_output_tensors  # TODO: output tensors should contain the policy version
            actions = preprocess_actions(self.env_info, self.policy_output_tensors['actions'])

        complete_rollouts, episodic_stats = [], []  # TODO

        with timing.add_time('env_step'):
            self.last_obs, rewards, dones, infos = self.vec_env.step(actions)

        with timing.add_time('post_env_step'):
            self.policy_id_buffer.fill_(self.policy_id)

            # TODO: for vectorized envs we either have a dictionary of tensors (isaacgym_examples), or a list of dictionaries (i.e. swarm_rl quadrotors)
            # Need an adapter class so it's consistent, i.e. always a dict of tensors.
            # this should yield indices of inactive agents
            #
            # if infos:
            #     inactive_agents = [i for i, info in enumerate(infos) if not info.get('is_active', True)]
            #     self.policy_id_buffer[inactive_agents] = -1
            # TODO: batcher runner probably won't have inactive agent support for now.

            # record the results from the env step
            processed_rewards = self._process_rewards(rewards, infos, self.policy_output_tensors['values'])
            self.curr_step[:] = dict(rewards=processed_rewards, dones=dones, policy_id=self.policy_id_buffer)

            # reset next-step hidden states to zero if we encountered an episode boundary
            # not sure if this is the best practice, but this is what everybody seems to be doing
            not_done = (1.0 - self.curr_step['dones'].float()).unsqueeze(-1)
            self.last_rnn_state = self.policy_output_tensors['new_rnn_states'] * not_done

            stats = self._process_env_step(rewards, dones, infos)
            episodic_stats.extend(stats)

        self.rollout_step += 1

        with timing.add_time('finalize_trajectories'):
            if self.rollout_step == self.cfg.rollout:
                # finalize and serialize the trajectory if we have a complete rollout
                complete_rollouts = self._finalize_trajectories()
                self.rollout_step = 0
                # we will need to request a new trajectory buffer!
                self.curr_traj_slice = self.curr_traj = None

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
