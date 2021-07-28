import os
import random
import signal
import time
from collections import OrderedDict
from queue import Empty, Full

import numpy as np
import psutil
import torch
from gym.spaces import Discrete, Tuple
from torch.multiprocessing import Process as TorchProcess

from sample_factory.algorithms.appo.appo_utils import TaskType, make_env_func, set_gpus_for_process
from sample_factory.algorithms.appo.policy_manager import PolicyManager
from sample_factory.algorithms.appo.population_based_training import PbtTask
from sample_factory.algorithms.utils.spaces.discretized import Discretized
from sample_factory.envs.env_utils import set_reward_shaping, find_training_info_interface, set_training_info
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, AttrDict, memory_consumption_mb, join_or_kill, set_process_cpu_affinity, \
    set_attr_if_exists, safe_put, safe_put_many


def transform_dict_observations(observations):
    """Transform list of dict observations into a dict of lists."""
    obs_dict = dict()
    if isinstance(observations[0], (dict, OrderedDict)):
        for key in observations[0].keys():
            if not isinstance(observations[0][key], str):
                obs_dict[key] = [o[key] for o in observations]
    else:
        # handle flat observations also as dict
        obs_dict['obs'] = observations

    for key, x in obs_dict.items():
        obs_dict[key] = np.stack(x)

    return obs_dict


class ActorState:
    """
    State of a single actor (agent) in a multi-agent environment.
    Single-agent environments are treated as multi-agent with one agent for simplicity.
    """

    def __init__(
        self, cfg, env, worker_idx, split_idx, env_idx, agent_idx,
        shared_buffers, policy_output_tensors,
        pbt_reward_shaping, policy_mgr,
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

        self.shared_buffers = shared_buffers
        self.curr_traj_buffer_idx = self.curr_traj_buffer = None
        self.update_traj_buffer(shared_buffers.get_trajectory_buffers(num_buffers=1)[0])

        policy_outputs_info = shared_buffers.policy_outputs
        self.policy_output_names = [p.name for p in policy_outputs_info]
        self.policy_output_sizes = [p.size for p in policy_outputs_info]
        self.policy_output_indices = np.cumsum(self.policy_output_sizes)
        self.policy_output_tensors = policy_output_tensors

        self.last_actions = None
        self.last_policy_steps = None
        self.last_rnn_state = None

        self.ready = False  # whether this agent received actions from the policy and can act in the environment again

        # By returning info = {'is_active': False, ...} the environment can indicate that the agent is not active,
        # i.e. dead or otherwise disabled. Experience from such agents will be ignored.
        self.is_active = True

        # This flag is reset at the beginning of every rollout. If the agent was inactive during the entire rollout
        # then it is ignored and no experience is sent to the learner.
        self.has_rollout_data = False

        self.needs_buffer = False  # whether this actor requires a new trajectory buffer

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

        self.integer_actions = False
        if isinstance(env.action_space, (Discrete, Discretized)):
            self.integer_actions = True
        if isinstance(env.action_space, Tuple):
            all_subspaces_discrete = all(isinstance(s, (Discrete, Discretized)) for s in env.action_space.spaces)
            if all_subspaces_discrete:
                self.integer_actions = True
            else:
                # tecnhically possible to add support for such spaces, but it's untested
                # for now, look at Discretized instead.
                raise Exception(
                    'Mixed discrete & continuous action spaces are not supported (should be an easy fix)'
                )

        self.env_training_info_interface = find_training_info_interface(env)

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
        self._reset_rnn_state()

        if self.cfg.with_pbt and self.pbt_reward_shaping[self.curr_policy_id] is not None:
            set_reward_shaping(self.env, self.pbt_reward_shaping[self.curr_policy_id], self.agent_idx)
            set_training_info(self.env_training_info_interface, self.approx_env_steps.get(self.curr_policy_id, 0))

        self._env_set_curr_policy()

    def update_traj_buffer(self, traj_buffer_idx):
        """Set ActorState to use a new shared buffer for the next trajectory."""
        self.curr_traj_buffer_idx = traj_buffer_idx
        self.curr_traj_buffer = self.shared_buffers.tensors.index(self.curr_traj_buffer_idx)
        self.needs_buffer = False

    def set_trajectory_data(self, data, rollout_step):
        """
        Write a dictionary of data into a trajectory buffer at the specific location (rollout_step).

        :param data: any sub-dictionary of the full per-step data, e.g. just observation, observation and action, etc.
        :param rollout_step: number of steps since we started the current rollout. When this reaches cfg.rollout
        we finalize the trajectory buffer and send it to the learner.
        """

        self.curr_traj_buffer.set_data(rollout_step, data)

    def _reset_rnn_state(self):
        self.last_rnn_state[:] = 0.0

    def curr_actions(self):
        """
        :return: the latest set of actions for this actor, calculated by the policy worker for the last observation
        """
        if self.integer_actions:
            actions = self.last_actions.astype(np.int32)
        else:
            actions = self.last_actions

        if len(actions) == 1:
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
            last_episode_true_reward = info.get('true_reward', self.last_episode_reward)
            last_episode_extra_stats = info.get('episode_extra_stats', dict())

            report = self.episodic_stats(last_episode_true_reward, last_episode_extra_stats)

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

            traj_buffer_idx = self.curr_traj_buffer_idx
            if traj_buffer_idx in buffers_used:
                # This rollout needs to be sent to multiple learners, i.e. because the policy changed in the middle
                # of the rollout. If we use the same shared buffer on multiple learners, we need some mechanism
                # to guarantee that this buffer will only be released once. It seems easier to just copy all data to
                # a new buffer for each additional learner. This should be a very rare event so the performance impact
                # is negligible.
                traj_buffer_idx = self.shared_buffers.get_trajectory_buffers(num_buffers=1)[0]
                buffer = self.shared_buffers.tensors.index(traj_buffer_idx)
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
            self._reset_rnn_state()

    def episodic_stats(self, last_episode_true_reward, last_episode_extra_stats):
        stats = dict(reward=self.last_episode_reward, len=self.last_episode_duration)

        stats['true_reward'] = last_episode_true_reward
        stats['episode_extra_stats'] = last_episode_extra_stats

        report = dict(episodic=stats, policy_id=self.curr_policy_id)
        self.last_episode_reward = self.last_episode_duration = 0
        return report


class VectorEnvRunner:
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

    def __init__(self, cfg, num_envs, worker_idx, split_idx, num_agents, shared_buffers, pbt_reward_shaping):
        """
        Ctor.

        :param cfg: global system config (all CLI params)
        :param num_envs: number of envs to run in this vector runner
        :param worker_idx: idx of the parent worker
        :param split_idx: index of the environment group in double-buffered sampling (either 0 or 1). Always 0 when
        double-buffered sampling is disabled.
        :param num_agents: number of agents in each env (1 for single-agent envs)
        :param shared_buffers: a collection of all shared data structures used by the algorithm. Most importantly,
        the trajectory buffers in shared memory.
        :param pbt_reward_shaping: initial reward shaping dictionary, for configuration where PBT optimizes
        reward coefficients in environments.
        """

        self.cfg = cfg

        self.num_envs = num_envs
        self.worker_idx = worker_idx
        self.split_idx = split_idx

        self.rollout_step = 0

        self.num_agents = num_agents  # queried from env

        self.shared_buffers = shared_buffers
        self.policy_output_tensors = self.shared_buffers.policy_output_tensors[self.worker_idx, self.split_idx]

        self.envs, self.actor_states, self.episode_rewards = [], [], []

        self.pbt_reward_shaping = pbt_reward_shaping

        self.policy_mgr = PolicyManager(self.cfg, self.num_agents)

    def init(self):
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
            env = make_env_func(self.cfg, env_config=env_config)

            env.seed(env_id)
            self.envs.append(env)

            actor_states_env, episode_rewards_env = [], []
            for agent_idx in range(self.num_agents):
                actor_state = ActorState(
                    self.cfg, env, self.worker_idx, self.split_idx, env_i, agent_idx,
                    self.shared_buffers, self.policy_output_tensors[env_i, agent_idx],
                    self.pbt_reward_shaping, self.policy_mgr,
                )
                actor_states_env.append(actor_state)
                episode_rewards_env.append(0.0)

            self.actor_states.append(actor_states_env)
            self.episode_rewards.append(episode_rewards_env)

    def update_env_steps(self, env_steps):
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                self.actor_states[env_i][agent_i].approx_env_steps = env_steps

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
                    new_rnn_state = None
                    for tensor_idx, name in enumerate(actor_state.policy_output_names):
                        if name == 'rnn_states':
                            new_rnn_state = policy_outputs[tensor_idx]
                        else:
                            policy_outputs_dict[name] = policy_outputs[tensor_idx]

                    # save parsed trajectory outputs directly into the trajectory buffer
                    actor_state.set_trajectory_data(policy_outputs_dict, self.rollout_step)
                    actor_state.last_actions = policy_outputs_dict['actions']

                    # this is an rnn state for the next iteration in the rollout
                    actor_state.last_rnn_state = new_rnn_state

                    actor_state.ready = True
                elif not actor_state.ready:
                    all_actors_ready = False

        # Potential optimization: when actions are ready for all actors within one environment we can execute
        # a simulation step right away, without waiting for all other actions to be calculated.
        return all_actors_ready

    def _process_rewards(self, rewards, env_i):
        """
        Pretty self-explanatory, here we record the episode reward and apply the optional clipping and
        scaling of rewards.
        """

        for agent_i, r in enumerate(rewards):
            self.actor_states[env_i][agent_i].last_episode_reward += r

        rewards = np.asarray(rewards, dtype=np.float32)
        rewards = rewards * self.cfg.reward_scale
        rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)
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

        rewards = self._process_rewards(rewards, env_i)

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
                rollouts.extend(self.actor_states[env_i][agent_i].finalize_trajectory(self.rollout_step))

        return rollouts

    def _update_trajectory_buffers(self, timing):
        """
        Request free trajectory buffers to store the next rollout.
        """
        num_buffers_to_request = 0
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                num_buffers_to_request += self.actor_states[env_i][agent_i].needs_buffer

        if num_buffers_to_request > 0:
            traj_buffer_indices = self.shared_buffers.get_trajectory_buffers(num_buffers_to_request, timing)

            i = 0
            for env_i in range(self.num_envs):
                for agent_i in range(self.num_agents):
                    actor_state = self.actor_states[env_i][agent_i]
                    if actor_state.needs_buffer:
                        buffer_idx = traj_buffer_indices[i]
                        actor_state.update_traj_buffer(buffer_idx)
                        i += 1

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

    def reset(self, report_queue):
        """
        Do the very first reset for all environments in a vector. Populate shared memory with initial obs.
        Note that this is called only once, at the very beginning of training. After this the envs should auto-reset.

        :param report_queue: we use report queue to monitor reset progress (see appo.py). This can be a lengthy
        process.
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
                actor_state.set_trajectory_data(dict(obs=obs), self.rollout_step)
                # rnn state is already initialized at zero

            safe_put(report_queue, dict(initialized_env=(self.worker_idx, self.split_idx, env_i)), queue_name='report')

        policy_request = self._format_policy_request()
        return policy_request

    def advance_rollouts(self, data, timing):
        """
        Main function in VectorEnvRunner. Does one step of simulation (if all actions for all actors are available).

        :param data: incoming data from policy workers (policy outputs), including new actions
        :param timing: this is just for profiling
        :return: same as reset(), return a set of requests for policy workers, asking them to generate actions for
        the next env step.
        """

        with timing.add_time('save_policy_outputs'):
            policy_id = data['policy_id']
            all_actors_ready = self._process_policy_outputs(policy_id, timing)
            if not all_actors_ready:
                # not all policies involved sent their actions, waiting for more
                return None, None, None

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
            self._update_trajectory_buffers(timing)
            self.rollout_step = 0

        with timing.add_time('prepare_next_step'):
            self._prepare_next_step()

        policy_request = self._format_policy_request()

        return policy_request, complete_rollouts, episodic_stats

    def close(self):
        for e in self.envs:
            e.close()


class ActorWorker:
    """
    Top-level class defining the actor worker (rollout worker in the paper)

    Works with an array (vector) of environments that is processed in portions.
    Simple case, env vector is split into two parts:
    1. Do an environment step in the 1st half of the vector (envs 1..N/2)
    2. Send observations to a queue for action generation elsewhere (e.g. on a GPU worker)
    3. Immediately start processing second half of the vector (envs N/2+1..N)
    4. By the time second half is processed, actions for the 1st half should be ready. Immediately start processing
    the 1st half of the vector again.

    As a result, if action generation is fast enough, this env runner should be busy 100% of the time
    calculating env steps, without waiting for actions.
    This is similar to double-buffered rendering in computer graphics.

    """

    def __init__(
        self, cfg, obs_space, action_space, num_agents, worker_idx, shared_buffers,
        task_queue, policy_queues, report_queue, learner_queues,
    ):
        """
        Ctor.

        :param cfg: global config (all CLI params)
        :param obs_space: observation space (spaces) of the environment
        :param action_space: action space(s)
        :param num_agents: number of agents per env (all env should have the same number of agents right now,
        although it should be easy to fix)
        :param worker_idx: index of this worker process
        :param shared_buffers: shared memory data structures initialized in main process (see shared_buffers.py)
        :param task_queue: queue for incoming messages for THIS particular actor worker. See the task types in the loop
        below, but the most common task is ROLLOUT_STEP, which means "here's your actions, advance simulation by
        one step".
        :param policy_queues: FIFO queues associated with all policies participating in training. We send requests
        for policy queue #N to get actions for envs (agents) that are controlled by policy #N.
        :param report_queue: one-way communication with the main process, various stats and whatnot
        :param learner_queues: one-way communication with the learner, sending trajectory buffers for learning
        """

        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_agents = num_agents

        self.worker_idx = worker_idx

        self.shared_buffers = shared_buffers

        self.terminate = False

        self.num_complete_rollouts = 0

        self.vector_size = cfg.num_envs_per_worker
        self.num_splits = cfg.worker_num_splits
        assert self.vector_size >= self.num_splits
        assert self.vector_size % self.num_splits == 0, 'Vector size should be divisible by num_splits'

        self.env_runners = None

        self.policy_queues = policy_queues
        self.report_queue = report_queue
        self.learner_queues = learner_queues
        self.task_queue = task_queue

        self.reward_shaping = [None for _ in range(self.cfg.num_policies)]

        self.process = TorchProcess(target=self._run, daemon=True)
        self.process.start()

    def _init(self):
        """
        Initialize env runners, that actually do all the work. Also we're doing some utility stuff here, e.g.
        setting process affinity (this is a performance optimization).
        """

        log.info('Initializing envs for env runner %d...', self.worker_idx)

        if self.cfg.force_envs_single_thread:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1, user_api=None)

        if self.cfg.set_workers_cpu_affinity:
            set_process_cpu_affinity(self.worker_idx, self.cfg.num_workers)
        psutil.Process().nice(min(self.cfg.default_niceness + 10, 20))

        self.env_runners = []
        for split_idx in range(self.num_splits):
            env_runner = VectorEnvRunner(
                self.cfg, self.vector_size // self.num_splits, self.worker_idx, split_idx, self.num_agents,
                self.shared_buffers, self.reward_shaping,
            )
            env_runner.init()
            self.env_runners.append(env_runner)

    def _terminate(self):
        for env_runner in self.env_runners:
            env_runner.close()

        self.terminate = True

    def _enqueue_policy_request(self, split_idx, policy_inputs):
        """Distribute action requests to their corresponding queues."""

        for policy_id, requests in policy_inputs.items():
            policy_request = (self.worker_idx, split_idx, requests)
            self.policy_queues[policy_id].put(policy_request)

        if not policy_inputs:
            # log.warning('No policy requests on worker %d-%d', self.worker_idx, split_idx)
            # log.warning('Send fake signal to our own queue to wake up the worker on the next iteration')
            advance_rollout_request = dict(split_idx=split_idx, policy_id=-1)
            self.task_queue.put((TaskType.ROLLOUT_STEP, advance_rollout_request))

    def _enqueue_complete_rollouts(self, complete_rollouts):
        """Send complete rollouts from VectorEnv to the learner."""
        if self.cfg.sampler_only:
            return

        rollouts_per_policy = dict()
        for rollout in complete_rollouts:
            policy_id = rollout['policy_id']
            if policy_id not in rollouts_per_policy:
                rollouts_per_policy[policy_id] = []

            rollouts_per_policy[policy_id].append(rollout)

        for policy_id, rollouts in rollouts_per_policy.items():
            self.learner_queues[policy_id].put((TaskType.TRAIN, rollouts))

    def _report_stats(self, stats):
        safe_put_many(self.report_queue, stats, queue_name='report')

    def _handle_reset(self):
        """
        Reset all envs, one split at a time (double-buffering), and send requests to policy workers to get
        actions for the very first env step.
        """
        for split_idx, env_runner in enumerate(self.env_runners):
            policy_inputs = env_runner.reset(self.report_queue)
            self._enqueue_policy_request(split_idx, policy_inputs)

        log.info('Finished reset for worker %d', self.worker_idx)
        safe_put(self.report_queue, dict(finished_reset=self.worker_idx), queue_name='report')

    def _advance_rollouts(self, data, timing):
        """
        Process incoming request from policy worker. Use the data (policy outputs, actions) to advance the simulation
        by one step on the corresponding VectorEnvRunner.

        If we successfully managed to advance the simulation, send requests to policy workers to get actions for the
        next step. If we completed the entire rollout, also send request to the learner!

        :param data: request from the policy worker, containing actions and other policy outputs
        :param timing: profiling stuff
        """
        split_idx = data['split_idx']

        runner = self.env_runners[split_idx]
        policy_request, complete_rollouts, episodic_stats = runner.advance_rollouts(data, timing)

        with timing.add_time('complete_rollouts'):
            if complete_rollouts:
                self._enqueue_complete_rollouts(complete_rollouts)

                if self.num_complete_rollouts == 0 and not self.cfg.benchmark:
                    # we just finished our first complete rollouts, perfect time to wait for experience derorrelation
                    # this guarantees that there won't be any obsolete trajectories when we awaken
                    delay = (float(self.worker_idx) / self.cfg.num_workers) * self.cfg.decorrelate_experience_max_seconds
                    log.info(
                        'Worker %d, sleep for %.3f sec to decorrelate experience collection',
                        self.worker_idx, delay,
                    )
                    time.sleep(delay)
                    log.info('Worker %d awakens!', self.worker_idx)

                self.num_complete_rollouts += len(complete_rollouts)

        with timing.add_time('enqueue_policy_requests'):
            if policy_request is not None:
                self._enqueue_policy_request(split_idx, policy_request)

        if episodic_stats:
            self._report_stats(episodic_stats)

    def _process_pbt_task(self, pbt_task):
        """Save the latest version of reward shaping from PBT, we later propagate this to envs."""
        task_type, data = pbt_task

        if task_type == PbtTask.UPDATE_REWARD_SCHEME:
            policy_id, new_reward_shaping_scheme = data
            self.reward_shaping[policy_id] = new_reward_shaping_scheme

    def _run(self):
        """
        Main loop of the actor worker (rollout worker).
        Process tasks (mainly ROLLOUT_STEP) until we get the termination signal, which usually means end of training.
        Currently there is no mechanism to restart dead workers if something bad happens during training. We can only
        retry on the initial reset(). This is definitely something to work on.
        """
        log.info('Initializing vector env runner %d...', self.worker_idx)
        log.info(f'ACTOR worker {self.worker_idx}\tpid {os.getpid()}\tparent {os.getppid()}')

        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if self.cfg.actor_worker_gpus:
            set_gpus_for_process(
                self.worker_idx,
                num_gpus_per_process=1, process_type='actor', gpu_mask=self.cfg.actor_worker_gpus,
            )

        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        last_report = time.time()
        with torch.no_grad():
            while not self.terminate:
                try:
                    try:
                        with timing.add_time('waiting'), timing.timeit('wait_actor'):
                            tasks = self.task_queue.get_many(timeout=0.1)
                    except Empty:
                        tasks = []

                    for task in tasks:
                        task_type, data = task

                        if task_type == TaskType.INIT:
                            self._init()
                            continue

                        if task_type == TaskType.TERMINATE:
                            self._terminate()
                            break

                        # handling actual workload
                        if task_type == TaskType.ROLLOUT_STEP:
                            if 'work' not in timing:
                                timing.waiting = 0  # measure waiting only after real work has started

                            with timing.add_time('work'), timing.timeit('one_step'):
                                self._advance_rollouts(data, timing)
                        elif task_type == TaskType.RESET:
                            with timing.add_time('reset'):
                                self._handle_reset()
                        elif task_type == TaskType.PBT:
                            self._process_pbt_task(data)
                        elif task_type == TaskType.UPDATE_ENV_STEPS:
                            for env in self.env_runners:
                                env.update_env_steps(data)

                    if time.time() - last_report > 5.0 and 'one_step' in timing:
                        timing_stats = dict(wait_actor=timing.wait_actor, step_actor=timing.one_step)
                        memory_mb = memory_consumption_mb()
                        stats = dict(memory_actor=memory_mb)
                        safe_put(self.report_queue, dict(timing=timing_stats, stats=stats), queue_name='report')
                        last_report = time.time()

                except RuntimeError as exc:
                    log.warning('Error while processing data w: %d, exception: %s', self.worker_idx, exc)
                    log.warning('Terminate process...')
                    self.terminate = True
                    safe_put(self.report_queue, dict(critical_error=self.worker_idx), queue_name='report')
                except KeyboardInterrupt:
                    self.terminate = True
                except:
                    log.exception('Unknown exception in rollout worker')
                    self.terminate = True

        if self.worker_idx <= 1:
            time.sleep(0.1)
            log.info(
                'Env runner %d, CPU aff. %r, rollouts %d: timing %s',
                self.worker_idx, psutil.Process().cpu_affinity(), self.num_complete_rollouts, timing,
            )

    def init(self):
        self.task_queue.put((TaskType.INIT, None))

    def request_reset(self):
        self.task_queue.put((TaskType.RESET, None))

    def request_step(self, split, actions):
        data = (split, actions)
        self.task_queue.put((TaskType.ROLLOUT_STEP, data))

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def update_env_steps(self, env_steps):
        try:
            self.task_queue.put_nowait((TaskType.UPDATE_ENV_STEPS, env_steps))
        except Full:
            pass

    def join(self):
        join_or_kill(self.process)
