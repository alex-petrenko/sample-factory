import random
from queue import Empty
from typing import Tuple, Dict, List, Optional

import numpy as np

from sample_factory.algo.sampling.sampling_utils import VectorEnvRunner
from sample_factory.algo.utils.tensor_dict import clone_tensor, ensure_numpy_array
from sample_factory.algorithms.appo.appo_utils import make_env_func_non_batched
from sample_factory.algorithms.appo.policy_manager import PolicyManager
from sample_factory.envs.env_utils import find_training_info_interface, set_reward_shaping, set_training_info
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import set_attr_if_exists, AttrDict, log


class ActorState:
    """
    State of a single actor (agent) in a multi-agent environment.
    Single-agent environments are treated as multi-agent with one agent for simplicity.
    """

    def __init__(
        self, cfg, env_info, env, worker_idx, split_idx, env_idx, agent_idx,
        buffer_mgr, traj_tensors, policy_output_tensors, pbt_reward_shaping, policy_mgr,
    ):
        self.cfg = cfg
        self.env = env
        self.worker_idx = worker_idx
        self.split_idx = split_idx
        self.env_idx = env_idx
        self.agent_idx = agent_idx

        self.policy_mgr = policy_mgr
        self.curr_policy_id = self.policy_mgr.get_policy_for_agent(agent_idx, env_idx)
        self._env_set_curr_policy()

        self.curr_traj_buffer_idx = self.curr_traj_buffer = None

        self.traj_tensors = traj_tensors

        self.policy_output_names = buffer_mgr.output_names
        self.policy_output_sizes = buffer_mgr.output_sizes
        self.policy_output_indices = np.cumsum(self.policy_output_sizes)[:-1]
        self.policy_output_tensors = policy_output_tensors

        self.last_actions = None
        self.last_policy_steps = None

        self.last_obs = None
        self.last_rnn_state = None
        self.last_value = None

        self.ready = False  # whether this agent received actions from the policy and can act in the environment again

        # By returning info = {'is_active': False, ...} the environment can indicate that the agent is not active,
        # i.e. dead or otherwise disabled. Experience from such agents will be ignored.
        self.is_active = True

        # This flag is reset at the beginning of every rollout. If the agent was inactive during the entire rollout
        # then it is ignored and no experience is sent to the learner.
        self.has_rollout_data = False

        self.needs_buffer = True  # whether this actor requires a new trajectory buffer

        self.num_trajectories = 0

        # have to count env steps per policy, since a single rollout may contain experience from more than one policy
        self.rollout_env_steps = {-1: 0}
        for policy_id in range(self.cfg.num_policies):
            self.rollout_env_steps[policy_id] = 0

        self.last_episode_reward = 0
        self.last_episode_duration = 0

        # dictionary with current training progress per policy
        # values are approximate because we get updates from the master process once every few seconds
        self.approx_env_steps = {}

        self.pbt_reward_shaping = pbt_reward_shaping

        self.integer_actions = env_info.integer_actions

        self.env_training_info_interface = find_training_info_interface(env)  # TODO: batched sampler

    def _env_set_curr_policy(self):
        """
        Most environments do not need to know index of the policy that currently collects experience.
        But in rare cases it is necessary. Originally was implemented for DMLab to properly manage the level cache.
        """
        set_attr_if_exists(self.env.unwrapped, 'curr_policy_idx', self.curr_policy_id)

    def _on_new_policy(self, new_policy_id):
        """Called when the new policy is sampled for this actor."""
        self.curr_policy_id = new_policy_id

        # policy change can only happen at the episode boundary so no need to reset rnn state (but I guess does not hurt)
        self.reset_rnn_state()

        if self.cfg.with_pbt and self.pbt_reward_shaping[self.curr_policy_id] is not None:
            set_reward_shaping(self.env, self.pbt_reward_shaping[self.curr_policy_id], self.agent_idx)
            set_training_info(self.env_training_info_interface, self.approx_env_steps.get(self.curr_policy_id, 0))

        self._env_set_curr_policy()

    def update_traj_buffer(self, traj_buffer_idx):
        """Set ActorState to use a new shared buffer for the next trajectory."""
        self.curr_traj_buffer_idx = traj_buffer_idx
        self.curr_traj_buffer = self.traj_tensors[self.curr_traj_buffer_idx]
        self.needs_buffer = False

    def set_trajectory_data(self, data: Dict, rollout_step: int):
        """
        Write a dictionary of data into a trajectory buffer at the specific location (rollout_step).

        :param data: any sub-dictionary of the full per-step data, e.g. just observation, observation and action, etc.
        :param rollout_step: number of steps since we started the current rollout. When this reaches cfg.rollout
        we finalize the trajectory buffer and send it to the learner.
        """

        self.curr_traj_buffer[rollout_step] = data

    def reset_rnn_state(self):
        self.last_rnn_state[:] = 0.0

    def curr_actions(self):
        """
        :return: the latest set of actions for this actor, calculated by the policy worker for the last observation
        """
        actions = ensure_numpy_array(self.last_actions)
        if self.integer_actions:
            actions = actions.astype(np.int32)

        if actions.ndim == 0:
            actions = actions.item()
        return actions

    def record_env_step(self, reward, done, info, rollout_step):
        """
        Policy inputs (obs) and policy outputs (actions, values, ...) for the current rollout step
        are already added to the trajectory buffer
        the only job remaining is to add auxiliary data: rewards, done flags, etc.

        :param reward: last reward from the env step
        :param done: last value of done flag
        :param info: info dictionary
        :param rollout_step: number of steps since we started the current rollout. When this reaches cfg.rollout
        we finalize the trajectory buffer and send it to the learner.
        """

        self.curr_traj_buffer['rewards'][rollout_step] = float(reward)
        self.curr_traj_buffer['dones'][rollout_step] = done

        policy_id = -1 if not self.is_active else self.curr_policy_id
        self.curr_traj_buffer['policy_id'][rollout_step] = policy_id

        self.has_rollout_data = self.has_rollout_data or self.is_active

        env_steps = info.get('num_frames', 1) if self.is_active else 0
        self.rollout_env_steps[policy_id] += env_steps
        self.last_episode_duration += env_steps

        self.is_active = info.get('is_active', True)

        report = None
        if done:
            last_episode_true_objective = info.get('true_objective', self.last_episode_reward)
            last_episode_extra_stats = info.get('episode_extra_stats', dict())

            report = self.episodic_stats(last_episode_true_objective, last_episode_extra_stats)

            set_training_info(self.env_training_info_interface, self.approx_env_steps.get(self.curr_policy_id, 0))

            new_policy_id = self.policy_mgr.get_policy_for_agent(self.agent_idx, self.env_idx)
            if new_policy_id != self.curr_policy_id:
                self._on_new_policy(new_policy_id)

        return report

    def finalize_trajectory(self, rollout_step):
        """
        Do some postprocessing after we finished the entire rollout.

        :param rollout_step: number of steps since we started the current rollout. This should be equal to
        cfg.rollout in this function
        :return: dictionary with auxiliary information about the trajectory
        """

        if not self.has_rollout_data:
            # this agent was inactive the entire rollout, send no trajectories to the learners
            return []

        self.has_rollout_data = False

        # Saving obs and hidden states for the step AFTER the last step in the current rollout.
        # We're going to need them later when we calculate next step value estimates.
        last_step_data = dict(obs=self.last_obs, rnn_states=self.last_rnn_state)
        self.set_trajectory_data(last_step_data, self.cfg.rollout)

        # We could change policy id in the middle of the rollout (i.e. on the episode boundary), in which case
        # this trajectory should be sent to two learners, one for the original policy id, one for the new one.
        # The part of the experience that belongs to a different policy will be ignored on the learner.
        trajectories = []
        buffers_used = set()

        for policy_id in np.unique(self.curr_traj_buffer['policy_id']):
            policy_id = int(policy_id)
            if policy_id == -1:
                # -1 is a policy that does not exist, used to mark inactive agents not controlled by any policy
                continue

            # TODO: do something about this crap
            traj_buffer_idx = self.curr_traj_buffer_idx
            if traj_buffer_idx in buffers_used:
                # This rollout needs to be sent to multiple learners, i.e. because the policy changed in the middle
                # of the rollout. If we use the same shared buffer on multiple learners, we need some mechanism
                # to guarantee that this buffer will only be released once. It seems easier to just copy all data to
                # a new buffer for each additional learner. This should be a very rare event so the performance impact
                # is negligible.
                traj_buffer_idx = self.buffer_mgr.get_trajectory_buffers(num_buffers=1)[0]
                buffer = self.buffer_mgr.tensors.index(traj_buffer_idx)
                buffer.set_data(slice(None), self.curr_traj_buffer)  # copy TensorDict data recursively

            buffers_used.add(traj_buffer_idx)

            t_id = f'{policy_id}_{self.worker_idx}_{self.split_idx}_{self.env_idx}_{self.agent_idx}_{self.num_trajectories}'
            traj_dict = dict(
                t_id=t_id, length=rollout_step, env_steps=self.rollout_env_steps[policy_id], policy_id=policy_id,
                traj_buffer_idx=traj_buffer_idx,
            )
            trajectories.append(traj_dict)
            self.num_trajectories += 1

        # reset rollout lengths
        self.rollout_env_steps = dict.fromkeys(self.rollout_env_steps, 0)

        assert buffers_used, 'We ought to send our buffer to at least one learner'
        self.needs_buffer = True

        return trajectories

    def update_rnn_state(self, done):
        """If we encountered an episode boundary, reset rnn states to their default values."""
        if done:
            self.reset_rnn_state()

    def episodic_stats(self, last_episode_true_objective, last_episode_extra_stats):
        stats = dict(reward=self.last_episode_reward, len=self.last_episode_duration)

        stats['true_objective'] = last_episode_true_objective
        stats['episode_extra_stats'] = last_episode_extra_stats

        report = dict(episodic=stats, policy_id=self.curr_policy_id)
        self.last_episode_reward = self.last_episode_duration = 0
        return report


class NonBatchedVectorEnvRunner(VectorEnvRunner):
    """
    A collection of environments simulated sequentially.
    With double buffering each actor worker holds two vector runners and switches between them.
    Without double buffering we only use a single VectorEnvRunner per actor worker.

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
            buffer_mgr, sampling_device: str, pbt_reward_shaping,  # TODO pbt reward
    ):
        """
        Ctor.

        :param cfg: global system config (all CLI params)
        :param num_envs: number of envs to run in this vector runner
        :param worker_idx: idx of the parent worker
        :param split_idx: index of the environment group in double-buffered sampling (either 0 or 1). Always 0 when
        double-buffered sampling is disabled.
        the trajectory buffers in shared memory.
        :param pbt_reward_shaping: initial reward shaping dictionary, for configuration where PBT optimizes
        reward coefficients in environments.
        """
        super().__init__(cfg, env_info, worker_idx, split_idx, buffer_mgr, sampling_device)

        self.num_envs = num_envs
        self.num_agents = env_info.num_agents

        self.envs, self.episode_rewards = [], []
        self.actor_states: List[List[ActorState]] = []

        self.need_trajectory_buffers = self.num_envs * self.num_agents

        self.pbt_reward_shaping = pbt_reward_shaping

        self.policy_mgr = PolicyManager(self.cfg, self.num_agents)

    def init(self, timing) -> Dict:
        """
        Actually instantiate the env instances.
        Also creates ActorState objects that hold the state of individual actors in (potentially) multi-agent envs.
        """

        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i

            # global env id within the entire system
            env_id = self.worker_idx * self.cfg.num_envs_per_worker + vector_idx

            env_config = AttrDict(
                worker_index=self.worker_idx, vector_index=vector_idx, env_id=env_id,
            )

            # log.info('Creating env %r... %d-%d-%d', env_config, self.worker_idx, self.split_idx, env_i)
            env = make_env_func_non_batched(self.cfg, env_config=env_config)

            env.seed(env_id)
            self.envs.append(env)

            actor_states_env, episode_rewards_env = [], []
            for agent_idx in range(self.num_agents):
                actor_state = ActorState(
                    self.cfg, self.env_info, env, self.worker_idx, self.split_idx, env_i, agent_idx,
                    self.buffer_mgr, self.traj_tensors, self.policy_output_tensors[env_i, agent_idx],
                    self.pbt_reward_shaping, self.policy_mgr,
                )
                actor_states_env.append(actor_state)
                episode_rewards_env.append(0.0)

            self.actor_states.append(actor_states_env)
            self.episode_rewards.append(episode_rewards_env)

        self.update_trajectory_buffers(timing, block=True)
        assert self.need_trajectory_buffers == 0

        policy_request = self._reset(timing)
        return policy_request

    # TODO: implement this on the runner
    def update_env_steps(self, env_steps):
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                self.actor_states[env_i][agent_i].approx_env_steps = env_steps

    def _reset(self, timing) -> Dict:
        """
        Do the very first reset for all environments in a vector. Populate shared memory with initial obs.
        Note that this is called only once, at the very beginning of training. After this the envs should auto-reset.

        :return: first requests for policy workers (to generate actions for the very first env step)
        """

        for env_i, e in enumerate(self.envs):
            observations = e.reset()

            if self.cfg.decorrelate_envs_on_one_worker:
                env_i_split = self.num_envs * self.split_idx + env_i
                decorrelate_steps = self.cfg.rollout * env_i_split + self.cfg.rollout * random.randint(0, 4)

                log.info('Decorrelating experience for %d frames...', decorrelate_steps)
                for decorrelate_step in range(decorrelate_steps):
                    actions = [e.action_space.sample() for _ in range(self.num_agents)]
                    observations, rew, dones, info = e.step(actions)

            for agent_i, obs in enumerate(observations):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.last_obs = obs
                actor_state.last_rnn_state = clone_tensor(self.traj_tensors['rnn_states'][actor_state.curr_traj_buffer_idx, 0])
                actor_state.reset_rnn_state()

        self.env_step_ready = True
        policy_request = self.generate_policy_request(timing)
        assert policy_request is not None
        return policy_request

    def _process_policy_outputs(self, policy_id, timing):
        """
        Process the latest data from the policy worker (for policy = policy_id).
        Policy outputs currently include new RNN states, actions, values, logprobs, etc. See shared_buffers.py
        for the full list of outputs.

        As a performance optimization, all these tensors are squished together into a single tensor.
        This allows us to copy them to shared memory only once, which makes a difference on the policy worker.
        Here we do np.split to separate them back into individual tensors.

        :param policy_id: index of the policy whose outputs we're currently processing
        :return: whether we got all outputs for all the actors in our VectorEnvRunner. If this is True then we're
        ready for the next step of the simulation.
        """

        all_actors_ready = True

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                if not actor_state.is_active:
                    continue

                actor_policy = actor_state.curr_policy_id
                assert actor_policy != -1

                if actor_policy == policy_id:
                    # via shared memory mechanism the new data should already be copied into the shared tensors

                    with timing.add_time('split_output_tensors'):
                        policy_outputs = np.split(
                            actor_state.policy_output_tensors,
                            indices_or_sections=actor_state.policy_output_indices,
                            axis=0,
                        )
                    policy_outputs_dict = dict()
                    for tensor_idx, name in enumerate(actor_state.policy_output_names):
                        policy_outputs_dict[name] = policy_outputs[tensor_idx].squeeze()

                    # save parsed trajectory outputs directly into the trajectory buffer
                    actor_state.set_trajectory_data(policy_outputs_dict, self.rollout_step)
                    actor_state.last_actions = policy_outputs_dict['actions']

                    # this is an rnn state for the next iteration in the rollout
                    actor_state.last_rnn_state = policy_outputs_dict['new_rnn_states']
                    actor_state.last_value = policy_outputs_dict['values'].item()

                    actor_state.ready = True
                elif not actor_state.ready:
                    all_actors_ready = False

        # Potential optimization: when actions are ready for all actors within one environment we can execute
        # a simulation step right away, without waiting for all other actions to be calculated.
        return all_actors_ready

    def _process_rewards(self, rewards, infos: List[Dict], env_i):
        """
        Pretty self-explanatory, here we record the episode reward and apply the optional clipping and
        scaling of rewards.
        """
        for agent_i, r in enumerate(rewards):
            self.actor_states[env_i][agent_i].last_episode_reward += r

        rewards = np.asarray(rewards, dtype=np.float32)
        rewards = rewards * self.cfg.reward_scale
        rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)

        if self.cfg.value_bootstrap:
            is_time_out = np.array([info.get('time_out', False) for info in infos])
            if any(is_time_out):
                # What we really want here is v(t+1) which we don't have, using v(t) is an approximation that
                # requires that rew(t) can be generally ignored.
                values = np.array([s.last_value for s in self.actor_states[env_i]])
                rewards += self.cfg.gamma * values * is_time_out

        return rewards

    def _process_env_step(self, new_obs, rewards, dones, infos, env_i):
        """
        Process step outputs from a single environment in the vector.

        :param new_obs: latest observations from the env
        :param env_i: index of the environment in the vector
        :return: episodic stats, not empty only on the episode boundary
        """

        episodic_stats = []
        env_actor_states = self.actor_states[env_i]

        rewards = self._process_rewards(rewards, infos, env_i)

        for agent_i in range(self.num_agents):
            actor_state = env_actor_states[agent_i]

            episode_report = actor_state.record_env_step(
                rewards[agent_i], dones[agent_i], infos[agent_i], self.rollout_step,
            )

            actor_state.last_obs = new_obs[agent_i]
            actor_state.update_rnn_state(dones[agent_i])

            if episode_report:
                episodic_stats.append(episode_report)

        return episodic_stats

    def _finalize_trajectories(self):
        """
        Do some postprocessing when we're done with the rollout.
        """

        rollouts = []
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor = self.actor_states[env_i][agent_i]
                rollouts.extend(actor.finalize_trajectory(self.rollout_step))
                self.need_trajectory_buffers += int(actor.needs_buffer)

        return rollouts

    def _format_policy_request(self):
        """
        Format data that allows us to request new actions from policies that control the agents in all the envs.
        Note how the data required is basically just indices of envs and agents, as well as location of the step
        data in the shared rollout buffer. This is enough for the policy worker to find the step data in the shared
        data structure.

        :return: formatted request to be distributed to policy workers through FIFO queues.
        """

        policy_request = dict()

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]

                if actor_state.is_active:
                    policy_id = actor_state.curr_policy_id

                    # where policy worker should look for the policy inputs for the next step
                    data = (env_i, agent_i, actor_state.curr_traj_buffer_idx, self.rollout_step)

                    if policy_id not in policy_request:
                        policy_request[policy_id] = []
                    policy_request[policy_id].append(data)

        return policy_request

    def _prepare_next_step(self):
        """
        Write environment outputs to shared memory so policy workers can calculate actions for the next step.
        Note how we temporarily hold obs and rnn_states in local variables before writing them into shared memory.
        We could not do the memory write right away because for that we need the memory location of the NEXT step.
        If this is the first step in the new rollout, we need to switch to a new trajectory buffer before we do that
        (because the previous trajectory buffer is now used by the learner and we can't use it until the learner is
        done).
        """

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]

                if actor_state.is_active:
                    actor_state.ready = False

                    # populate policy inputs in shared memory
                    policy_inputs = dict(obs=actor_state.last_obs, rnn_states=actor_state.last_rnn_state)
                    actor_state.set_trajectory_data(policy_inputs, self.rollout_step)
                else:
                    actor_state.ready = True

    def advance_rollouts(self, policy_id: PolicyID, timing) -> Tuple[List[Dict], List[Dict]]:
        """
        Main function in VectorEnvRunner. Does one step of simulation (if all actions for all actors are available).

        :param policy_id:
        :param timing: this is just for profiling
        :return: same as reset(), return a set of requests for policy workers, asking them to generate actions for
        the next env step.
        """

        with timing.add_time('save_policy_outputs'):
            all_actors_ready = self._process_policy_outputs(policy_id, timing)
            if not all_actors_ready:
                # not all policies involved sent their actions, waiting for more
                return [], []

        complete_rollouts, episodic_stats = [], []

        for env_i, e in enumerate(self.envs):
            with timing.add_time('env_step'):
                actions = [s.curr_actions() for s in self.actor_states[env_i]]
                new_obs, rewards, dones, infos = e.step(actions)

            with timing.add_time('overhead'):
                stats = self._process_env_step(new_obs, rewards, dones, infos, env_i)
                episodic_stats.extend(stats)

        self.rollout_step += 1
        if self.rollout_step == self.cfg.rollout:
            # finalize and serialize the trajectory if we have a complete rollout
            complete_rollouts = self._finalize_trajectories()
            self.rollout_step = 0

        self.env_step_ready = True
        return complete_rollouts, episodic_stats

    def update_trajectory_buffers(self, timing, block=False) -> bool:
        """
        Request free trajectory buffers to store the next rollout.
        """
        while self.need_trajectory_buffers > 0:
            with timing.add_time('wait_for_trajectories'):
                try:
                    buffers = self.traj_buffer_queue.get_many(block=block, max_messages_to_get=self.need_trajectory_buffers, timeout=1e9)
                    i = 0
                    for env_i in range(self.num_envs):
                        for agent_i in range(self.num_agents):
                            if i >= len(buffers):
                                break

                            actor_state = self.actor_states[env_i][agent_i]
                            if actor_state.needs_buffer:
                                buffer_idx = buffers[i]
                                actor_state.update_traj_buffer(buffer_idx)
                                self.need_trajectory_buffers -= 1
                                i += 1
                except Empty:
                    return False

        assert self.need_trajectory_buffers == 0
        return True

    def generate_policy_request(self, timing) -> Optional[Dict]:
        if not self.env_step_ready:
            # we haven't actually simulated the environment yet
            # log.debug('Cannot generate policy request because we have not finished the env simulation step yet!')
            return None

        if self.need_trajectory_buffers > 0:
            # we don't have a shared buffers to store data in - still waiting for one to become available
            return None

        with timing.add_time('prepare_next_step'):
            self._prepare_next_step()
            policy_request = self._format_policy_request()
            self.env_step_ready = False

        return policy_request

    def close(self):
        for e in self.envs:
            e.close()
