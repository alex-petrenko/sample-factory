import random
import signal
import time
from collections import OrderedDict
from queue import Empty

import numpy as np
import psutil
import torch
from torch.multiprocessing import Process as TorchProcess

from algorithms.appo.appo_utils import TaskType, cores_for_worker_process, make_env_func
from algorithms.appo.population_based_training import PbtTask
from algorithms.utils.algo_utils import num_env_steps
from utils.timing import Timing
from utils.utils import log, AttrDict, memory_consumption_mb, join_or_kill


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


# noinspection PyProtectedMember
def update_reward_shaping_scheme(env, agent_idx, reward_shaping):
    """
    Sending the updated reward shaping scheme to the environment, whether it's a single-agent env or a multi-agent env.
    """

    if hasattr(env.unwrapped, '_reward_shaping_wrapper'):
        env.unwrapped._reward_shaping_wrapper.reward_shaping_scheme = reward_shaping
    else:
        try:
            from envs.doom.multiplayer.doom_multiagent_wrapper import MultiAgentEnv
            if isinstance(env.unwrapped, MultiAgentEnv):
                env.unwrapped.set_env_attr(
                    agent_idx, 'unwrapped._reward_shaping_wrapper.reward_shaping_scheme', reward_shaping,
                )
        except ImportError:
            pass


class ActorState:
    """State of a single actor in an environment."""

    def __init__(
        self, cfg, env, worker_idx, split_idx, env_idx, agent_idx,
        traj_tensors, num_traj_buffers,
        policy_outputs_info, policy_output_tensors,
        pbt_reward_shaping,
    ):
        self.cfg = cfg
        self.env = env
        self.worker_idx = worker_idx
        self.split_idx = split_idx
        self.env_idx = env_idx
        self.agent_idx = agent_idx

        self.curr_policy_id = self._sample_random_policy()

        self.traj_tensors = traj_tensors
        self.num_traj_buffers = num_traj_buffers

        self.policy_output_names = [p.name for p in policy_outputs_info]
        self.policy_output_sizes = [p.size for p in policy_outputs_info]
        self.policy_output_tensors = policy_output_tensors

        self.last_actions = None
        self.last_rnn_state = None

        self.ready = False

        self.num_trajectories = 0
        self.rollout_env_steps = 0

        self.last_episode_reward = 0
        self.last_episode_duration = 0
        self.last_episode_true_reward = 0
        self.last_episode_extra_stats = dict()

        # whether the new episode was started during the current rollout
        self.new_episode = False

        self.pbt_reward_shaping = pbt_reward_shaping

    def _sample_random_policy(self):
        return random.randint(0, self.cfg.num_policies - 1)

    def _on_new_policy(self, new_policy_id):
        """Called when the new policy is sampled for this actor."""
        self.curr_policy_id = new_policy_id
        # we're switching to a different policy - reset the rnn hidden state
        self._reset_rnn_state()

        if self.cfg.with_pbt and self.pbt_reward_shaping[self.curr_policy_id] is not None:
            update_reward_shaping_scheme(self.env, self.agent_idx, self.pbt_reward_shaping[self.curr_policy_id])

    def set_trajectory_data(self, data, traj_buffer_idx, rollout_step):
        index = (traj_buffer_idx, rollout_step)
        self.traj_tensors.set_data(index, data)

    def _reset_rnn_state(self):
        self.last_rnn_state.fill_(0.0)

    def curr_actions(self):
        actions = self.last_actions.type(torch.int32).numpy()
        if len(actions) == 1:
            actions = actions.item()
        return actions

    def record_env_step(self, reward, done, info, traj_buffer_idx, rollout_step):
        # policy inputs (obs) and policy outputs (actions, values, ...) for the current rollout step
        # are already added to the trajectory buffer
        # the only job remaining is to add auxiliary data: rewards, done flags, etc.

        self.traj_tensors['rewards'][traj_buffer_idx, rollout_step][0] = float(reward)
        self.traj_tensors['dones'][traj_buffer_idx, rollout_step][0] = done

        env_steps = info.get('num_frames', 1)
        self.rollout_env_steps += env_steps
        self.last_episode_duration += env_steps

        if done:
            self.new_episode = True
            self.last_episode_true_reward = info.get('true_reward', self.last_episode_reward)
            self.last_episode_extra_stats = info.get('episode_extra_stats', dict())

    def finalize_trajectory(self, rollout_step):
        t_id = f'{self.curr_policy_id}_{self.worker_idx}_{self.split_idx}_{self.env_idx}_{self.agent_idx}_{self.num_trajectories}'
        traj_dict = dict(
            t_id=t_id, length=rollout_step, env_steps=self.rollout_env_steps, policy_id=self.curr_policy_id,
        )

        self.num_trajectories += 1
        self.rollout_env_steps = 0

        if self.new_episode:
            new_policy_id = self._sample_random_policy()
            if new_policy_id != self.curr_policy_id:
                self._on_new_policy(new_policy_id)

            self.new_episode = False

        return traj_dict

    def update_rnn_state(self, done):
        """If we encountered an episode boundary, reset rnn states to their default values."""
        if done:
            self._reset_rnn_state()

    def episodic_stats(self):
        stats = dict(reward=self.last_episode_reward, len=self.last_episode_duration)

        stats['true_reward'] = self.last_episode_true_reward
        stats['episode_extra_stats'] = self.last_episode_extra_stats

        report = dict(episodic=stats, policy_id=self.curr_policy_id)
        self.last_episode_reward = self.last_episode_duration = self.last_episode_true_reward = 0
        self.last_episode_extra_stats = dict()
        return report


class VectorEnvRunner:
    def __init__(self, cfg, num_envs, worker_idx, split_idx, num_agents, traj_buffers, pbt_reward_shaping):
        self.cfg = cfg

        self.num_envs = num_envs
        self.worker_idx = worker_idx
        self.split_idx = split_idx

        self.rollout_step = 0
        self.traj_buffer_idx = 0  # current shared trajectory buffer to use

        self.num_agents = num_agents  # queried from env

        index = (worker_idx, split_idx)
        self.traj_tensors = traj_buffers.tensors_individual_transitions.index(index)
        self.traj_tensors_available = traj_buffers.is_traj_tensor_available[index]
        self.num_traj_buffers = traj_buffers.num_traj_buffers
        self.policy_outputs = traj_buffers.policy_outputs
        self.policy_output_tensors = traj_buffers.policy_output_tensors[index]

        self.envs, self.actor_states, self.episode_rewards = [], [], []

        self.pbt_reward_shaping = pbt_reward_shaping

    def init(self):
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
                agent_traj_tensors = self.traj_tensors.index((env_i, agent_idx))
                actor_state = ActorState(
                    self.cfg, env, self.worker_idx, self.split_idx, env_i, agent_idx,
                    agent_traj_tensors, self.num_traj_buffers,
                    self.policy_outputs, self.policy_output_tensors[env_i, agent_idx],
                    self.pbt_reward_shaping,
                )
                actor_states_env.append(actor_state)
                episode_rewards_env.append(0.0)

            self.actor_states.append(actor_states_env)
            self.episode_rewards.append(episode_rewards_env)

    def _process_policy_outputs(self, policy_id):
        all_actors_ready = True

        for env_i in range(len(self.envs)):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                actor_policy = actor_state.curr_policy_id

                if actor_policy == policy_id:
                    # via shared memory mechanism the new data should already be copied into the shared tensors
                    policy_outputs = torch.split(
                        actor_state.policy_output_tensors,
                        split_size_or_sections=actor_state.policy_output_sizes,
                        dim=0,
                    )
                    policy_outputs_dict = dict()
                    new_rnn_state = None
                    for tensor_idx, name in enumerate(actor_state.policy_output_names):
                        if name == 'rnn_states':
                            new_rnn_state = policy_outputs[tensor_idx]
                        else:
                            policy_outputs_dict[name] = policy_outputs[tensor_idx]

                    # save parsed trajectory outputs directly into the trajectory buffer
                    actor_state.set_trajectory_data(policy_outputs_dict, self.traj_buffer_idx, self.rollout_step)
                    actor_state.last_actions = policy_outputs_dict['actions']

                    # this is an rnn state for the next iteration in the rollout
                    actor_state.last_rnn_state = new_rnn_state

                    actor_state.ready = True
                elif not actor_state.ready:
                    all_actors_ready = False

        # Potential optimization: when actions are ready for all actors within one environment we can execute
        # a simulation step right away, without waiting for all other actions to be calculated.
        # Can be useful when number of agents per environment is small.
        return all_actors_ready

    def _process_rewards(self, rewards, env_i):
        for agent_i, r in enumerate(rewards):
            self.actor_states[env_i][agent_i].last_episode_reward += r

        rewards = np.asarray(rewards, dtype=np.float32)
        rewards = rewards * self.cfg.reward_scale
        rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)
        return rewards

    def _process_env_step(self, new_obs, rewards, dones, infos, env_i):
        episodic_stats = []
        env_actor_states = self.actor_states[env_i]

        rewards = self._process_rewards(rewards, env_i)

        for agent_i in range(self.num_agents):
            actor_state = env_actor_states[agent_i]

            actor_state.record_env_step(
                rewards[agent_i], dones[agent_i], infos[agent_i], self.traj_buffer_idx, self.rollout_step,
            )

            actor_state.last_obs = new_obs[agent_i]
            actor_state.update_rnn_state(dones[agent_i])

            # save episode stats if we are at the episode boundary
            if dones[agent_i]:
                episodic_stats.append(actor_state.episodic_stats())

        return episodic_stats

    def _finalize_trajectories(self):
        rollouts = []
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                rollout = actor_state.finalize_trajectory(self.rollout_step)
                rollout['env_idx'] = env_i
                rollout['agent_idx'] = agent_i
                rollouts.append(rollout)

        return dict(rollouts=rollouts, traj_buffer_idx=self.traj_buffer_idx)

    def _format_policy_request(self):
        policy_request = dict()

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                policy_id = actor_state.curr_policy_id

                # where policy worker should look for the policy inputs for the next step
                data = (env_i, agent_i, self.traj_buffer_idx, self.rollout_step)

                if policy_id not in policy_request:
                    policy_request[policy_id] = []
                policy_request[policy_id].append(data)

        return policy_request

    def _prepare_next_step(self):
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.ready = False

                # populate policy inputs in shared memory
                policy_inputs = dict(obs=actor_state.last_obs, rnn_states=actor_state.last_rnn_state)
                actor_state.set_trajectory_data(policy_inputs, self.traj_buffer_idx, self.rollout_step)

    def rollout_tensors(self, traj_buff_idx):
        traj_buffers = dict()
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                traj_buffers[(env_i, agent_i)] = actor_state.trajectories[traj_buff_idx]

        return traj_buffers

    def reset(self, report_queue):
        for env_i, e in enumerate(self.envs):
            observations = e.reset()
            for agent_i, obs in enumerate(observations):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.set_trajectory_data(dict(obs=obs), self.traj_buffer_idx, self.rollout_step)
                # rnn state is already initialized at zero

            # log.debug(
            #     'Reset progress w:%d-%d finished %d/%d, still initializing envs...',
            #     self.worker_idx, self.split_idx, env_i + 1, len(self.envs),
            # )
            report_queue.put(dict(initialized_env=(self.worker_idx, self.split_idx, env_i)))

        policy_request = self._format_policy_request()
        return policy_request

    def advance_rollouts(self, data, timing):
        with timing.add_time('save_policy_outputs'):
            policy_id = data['policy_id']
            all_actors_ready = self._process_policy_outputs(policy_id)
            if not all_actors_ready:
                return None, None, None

        complete_rollouts, episodic_stats = [], []

        for env_i, e in enumerate(self.envs):
            with timing.add_time('env_step'):
                actions = [s.curr_actions() for s in self.actor_states[env_i]]
                new_obs, rewards, dones, infos = e.step(actions)

            with timing.add_time('overhead'):
                stats = self._process_env_step(new_obs, rewards, dones, infos, env_i)
                episodic_stats.extend(stats)

        self.rollout_step = self.rollout_step + 1
        if self.rollout_step == self.cfg.rollout:
            # finalize and serialize the trajectory if we have a complete rollout
            complete_rollouts = self._finalize_trajectories()
            self.rollout_step = 0
            self.traj_buffer_idx = (self.traj_buffer_idx + 1) % self.num_traj_buffers

            # wait for the next set of buffers to be released, if it's not ready yet
            # this should be a no-op, unless we are collecting experience faster than we can learn from it, in which
            # case this will act as a speed adjusting mechanism
            if self.traj_tensors_available[:, :, self.traj_buffer_idx].min() == 0:
                with timing.add_time('wait_buffers'):
                    self.wait_for_traj_buffers()

        self._prepare_next_step()
        policy_request = self._format_policy_request()

        return policy_request, complete_rollouts, episodic_stats

    def wait_for_traj_buffers(self):
        print_warning = True
        while self.traj_tensors_available[:, :, self.traj_buffer_idx].min() == 0:
            if print_warning:
                log.warning(
                    'Waiting for trajectory buffer %d on actor %d-%d',
                    self.traj_buffer_idx, self.worker_idx, self.split_idx,
                )
                print_warning = False
            time.sleep(0.002)

    def close(self):
        for e in self.envs:
            e.close()


class ActorWorker:
    """
    Works with an array (vector) of environments that is processes in portions.
    Simple case, env vector is split into two parts:
    1. Do an environment step in the 1st half of the vector (envs 1..N/2)
    2. Send observations to a queue for action generation elsewhere (e.g. on a GPU worker)
    3. Immediately start processing second half of the vector (envs N/2+1..N)
    4. By the time second half is processed, actions for the 1st half should be ready. Immediately start processing
    the 1st half of the vector again.

    As a result, if action generation is fast enough, this env runner should be busy 100% of the time
    calculating env steps, without waiting for actions.
    This is somewhat similar to double-buffered rendering in computer graphics.

    """

    def __init__(
        self, cfg, obs_space, action_space, num_agents, worker_idx, traj_buffers,
        task_queue, policy_queues, report_queue, learner_queues,
    ):
        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_agents = num_agents

        self.worker_idx = worker_idx

        self.traj_buffers = traj_buffers

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
        log.info('Initializing envs for env runner %d...', self.worker_idx)

        curr_process = psutil.Process()
        if self.cfg.set_workers_cpu_affinity:
            cpu_count = psutil.cpu_count()
            cores = cores_for_worker_process(self.worker_idx, self.cfg.num_workers, cpu_count)
            if cores is not None:
                curr_process.cpu_affinity(cores)

        log.debug('Worker %d uses CPU cores %r', self.worker_idx, curr_process.cpu_affinity())
        curr_process.nice(min(self.cfg.default_niceness + 10, 20))

        self.env_runners = []
        for split_idx in range(self.num_splits):
            env_runner = VectorEnvRunner(
                self.cfg, self.vector_size // self.num_splits, self.worker_idx, split_idx, self.num_agents,
                self.traj_buffers, self.reward_shaping,
            )
            env_runner.init()
            self.env_runners.append(env_runner)

    def _terminate(self):
        for env_runner in self.env_runners:
            env_runner.close()

        self.terminate = True

    def _enqueue_policy_request(self, split_idx, policy_inputs):
        for policy_id, requests in policy_inputs.items():
            policy_request = (self.worker_idx, split_idx, requests)
            self.policy_queues[policy_id].put(policy_request)

    def _enqueue_complete_rollouts(self, split_idx, complete_rollouts):
        """Send complete rollouts from VectorEnv to the learner."""
        if self.cfg.sampler_only:
            return

        rollouts = complete_rollouts['rollouts']
        traj_buffer_idx = complete_rollouts['traj_buffer_idx']

        # mark the trajectory buffer that we're sending to the learner as unavailable until the learner
        # finishes processing
        env_runner = self.env_runners[split_idx]
        env_runner.traj_tensors_available[:, :, traj_buffer_idx] = 0

        rollouts_per_policy = dict()
        for rollout in rollouts:
            policy_id = rollout['policy_id']
            if policy_id not in rollouts_per_policy:
                rollouts_per_policy[policy_id] = dict(
                    rollouts=[], worker_idx=self.worker_idx,
                    split_idx=split_idx, traj_buffer_idx=traj_buffer_idx,
                )

            rollouts_per_policy[policy_id]['rollouts'].append(rollout)

        for policy_id, rollouts in rollouts_per_policy.items():
            self.learner_queues[policy_id].put((TaskType.TRAIN, rollouts))

    def _report_stats(self, stats):
        for report in stats:
            self.report_queue.put(report)

    def _handle_reset(self):
        for split_idx, env_runner in enumerate(self.env_runners):
            policy_inputs = env_runner.reset(self.report_queue)
            self._enqueue_policy_request(split_idx, policy_inputs)

        log.info('Finished reset for worker %d', self.worker_idx)
        self.report_queue.put(dict(finished_reset=self.worker_idx))

    def _advance_rollouts(self, data, timing):
        split_idx = data['split_idx']

        runner = self.env_runners[split_idx]
        policy_request, complete_rollouts, episodic_stats = runner.advance_rollouts(data, timing)

        with timing.add_time('complete_rollouts'):
            if complete_rollouts:
                self._enqueue_complete_rollouts(split_idx, complete_rollouts)

                if self.num_complete_rollouts == 0 and not self.cfg.benchmark:
                    # we just finished our first complete rollouts, perfect time to wait for experience derorrelation
                    # this guarantees that there won't be any "old" trajectories when we awaken
                    delay = (float(self.worker_idx) / self.cfg.num_workers) * self.cfg.decorrelate_experience_max_seconds
                    log.info(
                        'Worker %d, sleep for %.3f sec to decorrelate experience collection',
                        self.worker_idx, delay,
                    )
                    time.sleep(delay)
                    log.info('Worker %d awakens!', self.worker_idx)

                self.num_complete_rollouts += len(complete_rollouts['rollouts'])

        with timing.add_time('enqueue_policy_requests'):
            if policy_request is not None:
                self._enqueue_policy_request(split_idx, policy_request)

        if episodic_stats:
            self._report_stats(episodic_stats)

    def _process_pbt_task(self, pbt_task):
        task_type, data = pbt_task

        if task_type == PbtTask.UPDATE_REWARD_SCHEME:
            policy_id, new_reward_shaping_scheme = data
            self.reward_shaping[policy_id] = new_reward_shaping_scheme

    def _run(self):
        log.info('Initializing vector env runner %d...', self.worker_idx)

        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

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

                    if time.time() - last_report > 5.0 and 'one_step' in timing:
                        timing_stats = dict(wait_actor=timing.wait_actor, step_actor=timing.one_step)
                        memory_mb = memory_consumption_mb()
                        stats = dict(memory_actor=memory_mb)
                        self.report_queue.put(dict(timing=timing_stats, stats=stats))
                        last_report = time.time()

                except RuntimeError as exc:
                    log.warning('Error while processing data w: %d, exception: %s', self.worker_idx, exc)
                    log.warning('Terminate process...')
                    self.terminate = True
                    self.report_queue.put(dict(critical_error=self.worker_idx))
                except KeyboardInterrupt:
                    self.terminate = True
                except:
                    log.exception('Unknown exception in rollout worker')
                    self.terminate = True

        if self.worker_idx <= 1:
            time.sleep(0.1)
            log.info('Env runner %d, rollouts %d: timing %s', self.worker_idx, self.num_complete_rollouts, timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))

    def request_reset(self):
        self.task_queue.put((TaskType.RESET, None))

    def request_step(self, split, actions):
        data = (split, actions)
        self.task_queue.put((TaskType.ROLLOUT_STEP, data))

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)

