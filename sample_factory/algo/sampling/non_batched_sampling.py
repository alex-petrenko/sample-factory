from __future__ import annotations

import random
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np

from sample_factory.algo.sampling.sampling_utils import VectorEnvRunner, record_episode_statistics_wrapper_stats
from sample_factory.algo.utils.agent_policy_mapping import AgentPolicyMapping
from sample_factory.algo.utils.env_info import EnvInfo, check_env_info
from sample_factory.algo.utils.make_env import make_env_func_non_batched
from sample_factory.algo.utils.misc import EPISODIC, POLICY_ID_KEY
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.algo.utils.tensor_dict import TensorDict, to_numpy
from sample_factory.algo.utils.tensor_utils import clone_tensor, ensure_numpy_array
from sample_factory.envs.env_utils import find_training_info_interface, set_reward_shaping, set_training_info
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import Config, MpQueue, PolicyID
from sample_factory.utils.utils import debug_log_every_n, log, set_attr_if_exists


class ActorState:
    """
    State of a single actor (agent) in a multi-agent environment.
    Single-agent environments are treated as multi-agent with one agent for simplicity.
    """

    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        env,
        worker_idx: int,
        split_idx: int,
        env_idx: int,
        agent_idx: int,
        global_env_idx: int,
        buffer_mgr: BufferMgr,
        traj_buffer_queue: MpQueue,
        traj_tensors: TensorDict,
        policy_output_tensors,
        training_info: List[Optional[Dict]],
        policy_mgr,
    ):
        self.cfg = cfg
        self.env = env
        self.env_info: EnvInfo = env_info

        self.worker_idx = worker_idx
        self.split_idx = split_idx
        self.env_idx = env_idx
        self.agent_idx = agent_idx
        self.global_env_idx: int = global_env_idx  # global index of the policy in the entire system

        self.policy_mgr = policy_mgr
        self.curr_policy_id = self.policy_mgr.get_policy_for_agent(agent_idx, env_idx, global_env_idx)
        self._env_set_curr_policy()

        self.curr_traj_buffer_idx: int = -4242424242  # uninitialized
        self.curr_traj_buffer: Optional[TensorDict] = None
        self.traj_tensors: TensorDict = traj_tensors

        self.buffer_mgr: BufferMgr = buffer_mgr
        self.traj_buffer_queue: MpQueue = traj_buffer_queue
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

        self.needs_buffer = True  # whether this actor requires a new trajectory buffer

        self.num_trajectories = 0

        self.last_episode_reward = 0
        self.last_episode_duration = 0

        self.training_info: List[Optional[Dict]] = training_info
        self.env_training_info_interface = find_training_info_interface(env)

    def _env_set_curr_policy(self):
        """
        Most environments do not need to know index of the policy that currently collects experience.
        But in rare cases it is necessary. Originally was implemented for DMLab to properly manage the level cache.
        """
        set_attr_if_exists(self.env.unwrapped, "curr_policy_idx", self.curr_policy_id)

    def _update_training_info(self) -> None:
        """Propagate information in the direction RL algo -> environment."""
        if self.training_info[self.curr_policy_id] is not None:
            reward_shaping = self.training_info[self.curr_policy_id].get("reward_shaping", None)
            set_reward_shaping(self.env, reward_shaping, self.agent_idx)
            set_training_info(self.env_training_info_interface, self.training_info[self.curr_policy_id])

    def _on_new_policy(self, new_policy_id):
        """Called when the new policy is sampled for this actor."""
        self.curr_policy_id = new_policy_id

        # policy change can only happen at the episode boundary so no need to reset rnn state (but I guess does not hurt)
        self.reset_rnn_state()

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

    def curr_actions(self) -> np.ndarray | List | Any:
        """
        :return: the latest set of actions for this actor, calculated by the policy worker for the last observation
        """
        actions = ensure_numpy_array(self.last_actions)

        if self.env_info.all_discrete or isinstance(self.env_info.action_space, gym.spaces.Discrete):
            return self._process_action_space(actions, is_discrete=True)
        elif isinstance(self.env_info.action_space, gym.spaces.Box):
            return self._process_action_space(actions, is_discrete=False)
        elif isinstance(self.env_info.action_space, gym.spaces.Tuple):
            out_actions = []
            for split, space in zip(
                np.split(actions, np.cumsum(self.env_info.action_splits)[:-1]), self.env_info.action_space
            ):
                is_discrete = isinstance(space, gym.spaces.Discrete)
                out_actions.append(self._process_action_space(split, is_discrete))
            return out_actions

        raise NotImplementedError(f"Unknown action space type: {type(self.env_info.action_space)}")

    @staticmethod
    def _process_action_space(actions: np.ndarray, is_discrete: bool) -> np.ndarray | Any:
        if is_discrete:
            actions = actions.astype(np.int32)
            if actions.size == 1:
                # this will turn a 1-element array into single Python scalar (int). Works for 0-D and 1-D arrays.
                actions = actions.item()
        else:
            if actions.ndim == 0:
                # envs with continuous actions typically expect a vector of actions (i.e. Mujoco)
                # if there's only one action (i.e. Mujoco pendulum) then we need to make it a 1D vector
                actions = np.expand_dims(actions, -1)

        return actions

    def record_env_step(self, reward, terminated: bool, truncated: bool, info, rollout_step):
        """
        Policy inputs (obs) and policy outputs (actions, values, ...) for the current rollout step
        are already added to the trajectory buffer
        the only job remaining is to add auxiliary data: rewards, done flags, etc.

        :param reward: last reward from the env step
        :param terminated: whether the episode has terminated
        :param truncated: whether the episode has been truncated (i.e. max episode length reached)
        :param info: info dictionary
        :param rollout_step: number of steps since we started the current rollout. When this reaches cfg.rollout
        we finalize the trajectory buffer and send it to the learner.
        """

        done = terminated | truncated

        self.curr_traj_buffer["rewards"][rollout_step] = float(reward)
        self.curr_traj_buffer["dones"][rollout_step] = done
        self.curr_traj_buffer["time_outs"][rollout_step] = truncated

        # -1 policy_id does not match any valid policy on the learner, therefore this will be treated as
        # invalid data coming from a different policy and should be ignored by the learner.
        policy_id = -1 if not self.is_active else self.curr_policy_id
        self.curr_traj_buffer["policy_id"][rollout_step] = policy_id

        # multiply by frameskip to get the episode lenghts matching the actual number of simulated steps
        self.last_episode_duration += self.env_info.frameskip if self.cfg.summaries_use_frameskip else 1

        self.is_active = info.get("is_active", True)

        report = None
        if done:
            report = self._episodic_stats(info)

            self._update_training_info()

            new_policy_id = self.policy_mgr.get_policy_for_agent(self.agent_idx, self.env_idx, self.global_env_idx)
            if new_policy_id != self.curr_policy_id:
                self._on_new_policy(new_policy_id)

            self.last_episode_reward = self.last_episode_duration = 0.0

        return report

    def finalize_trajectory(self, rollout_step: int) -> List[Dict[str, Any]]:
        """
        Do some postprocessing after we finished the entire rollout.

        :param rollout_step: number of steps since we started the current rollout. This should be equal to
        cfg.rollout in this function
        :return: dictionary with auxiliary information about the trajectory
        """

        # Saving obs and hidden states for the step AFTER the last step in the current rollout.
        # We're going to need them later when we calculate next step value estimates.
        last_step_data = dict(obs=self.last_obs, rnn_states=self.last_rnn_state)
        self.set_trajectory_data(last_step_data, self.cfg.rollout)

        # We could change policy id in the middle of the rollout (i.e. on the episode boundary), in which case
        # this trajectory should be sent to two learners, one for the original policy id, one for the new one.
        # The part of the experience that belongs to a different policy will be ignored on the learner.
        trajectories = []
        policy_buffers: Dict[PolicyID, int] = dict()

        unique_policies = np.unique(self.curr_traj_buffer["policy_id"])
        if len(unique_policies) > 1:
            debug_log_every_n(
                1000, f"Multiple policies in trajectory buffer: {unique_policies} (-1 means inactive agent)"
            )

        for policy_id in unique_policies:
            policy_id = int(policy_id)
            if policy_id == -1:
                # The entire trajectory belongs to an inactive agent, we send it to the current policy learner
                # the ideal solution would be to ditch this rollout entirely but this can mess with the
                # sync mode algorithm for counting how many trajectories we should advance at a time.
                # Learner will carefully mask the inactive (invalid) data so it should be okay to do this.
                policy_id = self.curr_policy_id

            if policy_id in policy_buffers:
                # we already created a request for this policy
                continue

            traj_buffer_idx = self.curr_traj_buffer_idx
            if traj_buffer_idx in policy_buffers.values():
                # This rollout needs to be sent to multiple learners, i.e. because the policy changed in the middle
                # of the rollout. If we use the same shared buffer on multiple learners, we need some mechanism
                # to guarantee that this buffer will only be released once. It seems easier to just copy all data to
                # a new buffer for each additional learner. This should be a very rare event so the performance impact
                # is negligible.
                try:
                    traj_buffer_idx = self.traj_buffer_queue.get(block=True, timeout=100)
                except Empty:
                    log.error(
                        f"Lost trajectory for {policy_id=} ({self.curr_traj_buffer['policy_id']}) since we could not find a trajectory buffer!"
                    )
                    continue

                buffer = self.traj_tensors[traj_buffer_idx]
                buffer[:] = self.curr_traj_buffer  # copy TensorDict data recursively

            policy_buffers[policy_id] = traj_buffer_idx

            t_id = f"{policy_id}_{self.worker_idx}_{self.split_idx}_{self.env_idx}_{self.agent_idx}_{self.num_trajectories}"
            traj_dict = dict(t_id=t_id, length=rollout_step, policy_id=policy_id, traj_buffer_idx=traj_buffer_idx)
            trajectories.append(traj_dict)
            self.num_trajectories += 1

        assert len(policy_buffers), "We ought to send our buffer to at least one learner"
        self.needs_buffer = True

        return trajectories

    def update_rnn_state(self, done):
        """If we encountered an episode boundary, reset rnn states to their default values."""
        if done:
            self.reset_rnn_state()

    def _episodic_stats(self, info: Dict) -> Dict[str, Any]:
        stats = dict(
            reward=self.last_episode_reward,
            len=self.last_episode_duration,
            episode_extra_stats=info.get("episode_extra_stats", dict()),
        )

        if (true_objective := info.get("true_objective", self.last_episode_reward)) is not None:
            stats["true_objective"] = true_objective

        episode_wrapper_stats = record_episode_statistics_wrapper_stats(info)
        if episode_wrapper_stats is not None:
            wrapper_rew, wrapper_len = episode_wrapper_stats
            stats["RecordEpisodeStatistics_reward"] = wrapper_rew
            stats["RecordEpisodeStatistics_len"] = wrapper_len

        report = {EPISODIC: stats, POLICY_ID_KEY: self.curr_policy_id}
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
        self,
        cfg,
        env_info,
        num_envs,
        worker_idx,
        split_idx,
        buffer_mgr,
        sampling_device: str,
        training_info: List[Optional[Dict[str, Any]]],
    ):
        """
        Ctor.

        :param cfg: global system config (all CLI params)
        :param num_envs: number of envs to run in this vector runner
        :param worker_idx: idx of the parent worker
        :param split_idx: index of the environment group in double-buffered sampling (either 0 or 1). Always 0 when
        double-buffered sampling is disabled.
        :param training_info: curr env steps, reward shaping scheme, etc.
        """
        super().__init__(cfg, env_info, worker_idx, split_idx, buffer_mgr, sampling_device)

        if sampling_device == "cpu":
            # TODO: comment
            self.traj_tensors = to_numpy(self.traj_tensors)
            self.policy_output_tensors = to_numpy(self.policy_output_tensors)

        self.num_envs = num_envs
        self.num_agents = env_info.num_agents

        self.envs, self.episode_rewards = [], []
        self.actor_states: List[List[ActorState]] = []

        self.need_trajectory_buffers = self.num_envs * self.num_agents

        self.training_info: List[Optional[Dict]] = training_info

        self.policy_mgr = AgentPolicyMapping(self.cfg, self.env_info)

    def init(self, timing: Timing):
        """
        Actually instantiate the env instances.
        Also creates ActorState objects that hold the state of individual actors in (potentially) multi-agent envs.
        """
        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i

            # global env id within the entire system
            global_env_idx = self.worker_idx * self.cfg.num_envs_per_worker + vector_idx

            env_config = AttrDict(
                worker_index=self.worker_idx,
                vector_index=vector_idx,
                env_id=global_env_idx,
            )

            # log.info('Creating env %r... %d-%d-%d', env_config, self.worker_idx, self.split_idx, env_i)
            env = make_env_func_non_batched(self.cfg, env_config=env_config)
            check_env_info(env, self.env_info, self.cfg)

            self.envs.append(env)

            actor_states_env, episode_rewards_env = [], []
            for agent_idx in range(self.num_agents):
                actor_state = ActorState(
                    self.cfg,
                    self.env_info,
                    env,
                    self.worker_idx,
                    self.split_idx,
                    env_i,
                    agent_idx,
                    global_env_idx,
                    self.buffer_mgr,
                    self.traj_buffer_queue,
                    self.traj_tensors,
                    self.policy_output_tensors[env_i, agent_idx],
                    self.training_info,
                    self.policy_mgr,
                )
                actor_states_env.append(actor_state)
                episode_rewards_env.append(0.0)

            self.actor_states.append(actor_states_env)
            self.episode_rewards.append(episode_rewards_env)

        self._reset()

    def _reset(self):
        """
        Do the very first reset for all environments in a vector. Populate shared memory with initial obs.
        Note that this is called only once, at the very beginning of training. After this the envs should auto-reset.

        :return: first requests for policy workers (to generate actions for the very first env step)
        """

        for env_i, e in enumerate(self.envs):
            seed = self.actor_states[env_i][0].global_env_idx
            observations, info = e.reset(seed=seed)  # new way of doing seeding since Gym 0.26.0

            if self.cfg.decorrelate_envs_on_one_worker:
                env_i_split = self.num_envs * self.split_idx + env_i
                decorrelate_steps = self.cfg.rollout * env_i_split

                log.info("Decorrelating experience for %d frames...", decorrelate_steps)
                for decorrelate_step in range(decorrelate_steps):
                    actions = [e.action_space.sample() for _ in range(self.num_agents)]
                    observations, rew, terminated, truncated, info = e.step(actions)

            for agent_i, obs in enumerate(observations):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.last_obs = obs
                actor_state.last_rnn_state = clone_tensor(self.traj_tensors["rnn_states"][0, 0])
                actor_state.reset_rnn_state()

        self.env_step_ready = True

    def _process_policy_outputs(self, policy_id, timing):
        """
        Process the latest data from the policy worker (for policy = policy_id).
        Policy outputs currently include new RNN states, actions, values, logprobs, etc.

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

                    with timing.add_time("split_output_tensors"):
                        policy_outputs = np.split(
                            actor_state.policy_output_tensors,
                            indices_or_sections=actor_state.policy_output_indices,
                            axis=0,
                        )
                    policy_outputs_dict = dict()
                    for tensor_idx, name in enumerate(actor_state.policy_output_names):
                        policy_outputs_dict[name] = policy_outputs[tensor_idx]

                    # save parsed trajectory outputs directly into the trajectory buffer
                    actor_state.set_trajectory_data(policy_outputs_dict, self.rollout_step)
                    actor_state.last_actions = policy_outputs_dict["actions"].squeeze()

                    # this is an rnn state for the next iteration in the rollout
                    actor_state.last_rnn_state = policy_outputs_dict["new_rnn_states"]
                    actor_state.last_value = policy_outputs_dict["values"].item()

                    actor_state.ready = True
                elif not actor_state.ready:
                    all_actors_ready = False

        # Potential optimization: when actions are ready for all actors within one environment we can execute
        # a simulation step right away, without waiting for all other actions to be calculated.
        return all_actors_ready

    def _process_rewards(self, rewards, env_i: int):
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

    def _process_env_step(self, new_obs, rewards, terminated, truncated, infos, env_i):
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
                rewards[agent_i],
                terminated[agent_i],
                truncated[agent_i],
                infos[agent_i],
                self.rollout_step,
            )

            actor_state.last_obs = new_obs[agent_i]
            actor_state.update_rnn_state(terminated[agent_i] | truncated[agent_i])

            if episode_report:
                episodic_stats.append(episode_report)

        return episodic_stats

    def _finalize_trajectories(self) -> List[Dict[str, Any]]:
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
        with timing.add_time("save_policy_outputs"):
            all_actors_ready = self._process_policy_outputs(policy_id, timing)
            if not all_actors_ready:
                # not all policies involved sent their actions, waiting for more
                return [], []

        complete_rollouts, episodic_stats = [], []

        for env_i, e in enumerate(self.envs):
            with timing.add_time("env_step"):
                actions = [s.curr_actions() for s in self.actor_states[env_i]]
                new_obs, rewards, terminated, truncated, infos = e.step(actions)

            with timing.add_time("overhead"):
                stats = self._process_env_step(new_obs, rewards, terminated, truncated, infos, env_i)
                episodic_stats.extend(stats)

        self.rollout_step += 1
        if self.rollout_step == self.cfg.rollout:
            # finalize and serialize the trajectory if we have a complete rollout
            complete_rollouts = self._finalize_trajectories()
            self.rollout_step = 0

        self.env_step_ready = True
        return complete_rollouts, episodic_stats

    def update_trajectory_buffers(self, timing) -> bool:
        """
        Request free trajectory buffers to store the next rollout.
        """
        while self.need_trajectory_buffers > 0:
            with timing.add_time("wait_for_trajectories"):
                try:
                    buffers = self.traj_buffer_queue.get_many(
                        block=False,
                        max_messages_to_get=self.need_trajectory_buffers,
                    )
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

    def generate_policy_request(self) -> Optional[Dict]:
        if not self.env_step_ready:
            # we haven't actually simulated the environment yet
            # log.debug('Cannot generate policy request because we have not finished the env simulation step yet!')
            return None

        if self.need_trajectory_buffers > 0:
            # we don't have a shared buffers to store data in - still waiting for one to become available
            return None

        self._prepare_next_step()
        policy_request = self._format_policy_request()
        self.env_step_ready = False
        return policy_request

    def synchronize_devices(self) -> None:
        """
        Non-batched sampling on GPU does not really make sense, so we currently leave this as a no-op.
        If in the future we want to do non-batched sampling with GPU-side observations, we should add synchronization
        here.
        """
        pass

    def close(self):
        for e in self.envs:
            e.close()
