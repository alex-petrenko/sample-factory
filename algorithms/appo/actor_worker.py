import copy
import queue
import random
import threading
import time
from collections import OrderedDict
from multiprocessing import Process

import numpy as np
import ray.pyarrow_files.pyarrow as pa
from ray.pyarrow_files.pyarrow import plasma

from algorithms.appo.appo_utils import TaskType
from algorithms.utils.algo_utils import num_env_steps
from algorithms.utils.multi_agent import MultiAgentWrapper
from algorithms.utils.multi_env import safe_get
from envs.create_env import create_env
from utils.timing import Timing
from utils.utils import log, AttrDict


def make_env_func(cfg, env_config):
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)
    if not hasattr(env, 'num_agents'):
        env = MultiAgentWrapper(env)
    return env


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
    """State of a single actor in an environment."""

    def __init__(self, cfg, worker_idx, split_idx, env_idx, agent_idx):
        self.cfg = cfg
        self.worker_idx = worker_idx
        self.split_idx = split_idx
        self.env_idx = env_idx
        self.agent_idx = agent_idx

        self.curr_policy_id = self._sample_random_policy()
        self.rnn_state = self._reset_rnn_state()
        self.last_obs = self.curr_actions = None
        self.ready = False

        self.trajectory = dict()
        self.num_trajectories = 0
        self.rollout_env_steps = 0

        self.last_episode_reward = 0
        self.last_episode_duration = 0

        # whether the new episode was started during the current rollout
        self.new_episode = False

    def _sample_random_policy(self):
        return random.randint(0, self.cfg.num_policies - 1)

    def _reset_rnn_state(self):
        return np.zeros([self.cfg.hidden_size], dtype=np.float32)

    def _trajectory_add_args(self, **kwargs):
        for arg_name, arg_value in kwargs.items():
            if arg_value is None:
                continue

            if arg_name not in self.trajectory:
                self.trajectory[arg_name] = [arg_value]
            else:
                self.trajectory[arg_name].append(arg_value)

    def trajectory_add_policy_inputs(self, obs, rnn_states):
        self._trajectory_add_args(obs=obs, rnn_states=rnn_states)

    def trajectory_add_env_step(self, reward, done, info):
        self._trajectory_add_args(rewards=reward, dones=done)

        env_steps = num_env_steps([info])
        self.rollout_env_steps += env_steps
        self.last_episode_duration += env_steps

        if done:
            self.new_episode = True

    def trajectory_add_policy_step(self, actions, action_logits, log_prob_actions, values, policy_version):
        args = copy.copy(locals())
        del args['self']  # only args passed to the function without "self"
        self._trajectory_add_args(**args)

    def trajectory_len(self):
        key = 'rewards'  # can be anything
        return len(self.trajectory[key]) if key in self.trajectory else 0

    def finalize_trajectory(self):
        obs_dict = transform_dict_observations(self.trajectory['obs'])
        self.trajectory['obs'] = obs_dict

        t_id = f'{self.curr_policy_id}_{self.worker_idx}_{self.split_idx}_{self.env_idx}_{self.agent_idx}_{self.num_trajectories}'
        traj_dict = dict(
            t_id=t_id, length=self.trajectory_len(), env_steps=self.rollout_env_steps, policy_id=self.curr_policy_id,
            t=self.trajectory,
        )

        self.trajectory = dict()
        self.num_trajectories += 1
        self.rollout_env_steps = 0

        if self.new_episode:
            new_policy_id = self._sample_random_policy()
            if new_policy_id != self.curr_policy_id:
                # we're switching to a different policy - reset the rnn hidden state
                self.curr_policy_id = new_policy_id
                self.rnn_state = self._reset_rnn_state()

            self.new_episode = False

        return traj_dict

    def update_rnn_state(self, done):
        if done:
            self.rnn_state = self._reset_rnn_state()

        return self.rnn_state

    def episodic_stats(self):
        stats = dict(reward=self.last_episode_reward, len=self.last_episode_duration)
        report = dict(episodic=stats, policy_id=self.curr_policy_id)
        self.last_episode_reward = self.last_episode_duration = 0
        return report


class VectorEnvRunner:
    def __init__(self, cfg, num_envs, worker_idx, split_idx, plasma_store):
        self.cfg = cfg

        self.num_envs = num_envs
        self.worker_idx = worker_idx
        self.split_idx = split_idx

        self.rollout_step = 0

        self.num_agents = -1  # queried from env

        self.envs, self.actor_states, self.episode_rewards = [], [], []

        self.plasma_client, self.serialization_context = plasma_store

    def init(self):
        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i
            env_config = AttrDict({'worker_index': self.worker_idx, 'vector_index': vector_idx})
            # log.info('Creating env %r... %d-%d-%d', env_config, self.worker_idx, self.split_idx, env_i)
            env = make_env_func(self.cfg, env_config=env_config)

            if not hasattr(env, 'num_agents'):
                env = MultiAgentWrapper(env)
            self.num_agents = env.num_agents

            env.seed(self.worker_idx * 1000 + env_i)
            self.envs.append(env)

            actor_states_env, episode_rewards_env = [], []
            for agent_idx in range(self.num_agents):
                actor_state = ActorState(self.cfg, self.worker_idx, self.split_idx, env_i, agent_idx)
                actor_states_env.append(actor_state)
                episode_rewards_env.append(0.0)

            self.actor_states.append(actor_states_env)
            self.episode_rewards.append(episode_rewards_env)

    def _save_policy_outputs(self, policy_id, policy_outputs, policy_version):
        policy_outputs = self.plasma_client.get(policy_outputs, -1, serialization_context=self.serialization_context)
        all_actors_ready = True

        i = 0
        for env_i in range(len(self.envs)):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                actor_policy = actor_state.curr_policy_id

                if actor_policy == policy_id:
                    outputs = AttrDict(policy_outputs)
                    actor_state.curr_actions = outputs.actions[i]
                    actor_state.rnn_state = outputs.rnn_states[i]
                    actor_state.trajectory_add_policy_step(
                        outputs.actions[i],
                        outputs.action_logits[i],
                        outputs.log_prob_actions[i],
                        outputs.values[i],
                        policy_version,
                    )

                    actor_state.ready = True

                    # number of actors with this policy id should match then number of policy outputs for this id
                    i += 1

                if not actor_state.ready:
                    all_actors_ready = False

        # Potential optimization: when actions are ready for all actors within one environment we can execute
        # a simulation step right away, without waiting for all other actions to be calculated.
        # Can be useful when number of agents per environment is small.
        return all_actors_ready

    def _process_rewards(self, rewards, env_i):
        for agent_i, r in enumerate(rewards):
            self.actor_states[env_i][agent_i].last_episode_reward += r

        rewards = np.asarray(rewards, dtype=np.float32)
        rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)
        rewards = rewards * self.cfg.reward_scale
        return rewards

    def _process_env_step(self, new_obs, rewards, dones, infos, env_i):
        complete_rollouts, episodic_stats = [], []
        env_actor_states = self.actor_states[env_i]

        rewards = self._process_rewards(rewards, env_i)

        for agent_i in range(self.num_agents):
            actor_state = env_actor_states[agent_i]

            # add information from the last env step to the trajectory, after this call
            # len(obs) == len(rnn_states) == len(rewards) == ...
            actor_state.trajectory_add_env_step(rewards[agent_i], dones[agent_i], infos[agent_i])

            # finalize and serialize the trajectory if we have a complete rollout
            if actor_state.trajectory_len() >= self.cfg.rollout:
                complete_rollouts.append(actor_state.finalize_trajectory())

            # if we encountered an episode boundary, reset rnn states to their default values
            new_rnn_state = actor_state.update_rnn_state(dones[agent_i])

            # save latest policy inputs (obs and hidden states)
            # after this block len(obs) == len(rnn_states) == len(rewards) + 1 == len(dones) + 1 == ...
            actor_state.last_obs = new_obs[agent_i]
            actor_state.trajectory_add_policy_inputs(actor_state.last_obs, new_rnn_state)

            # save episode stats if we are at the episode boundary
            if dones[agent_i]:
                # TODO! stats per policy
                episodic_stats.append(actor_state.episodic_stats())

        return complete_rollouts, episodic_stats

    def _format_policy_inputs(self):
        policy_inputs = dict()
        policy_num_inputs = [0 for _ in range(self.cfg.num_policies)]

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                policy_id = actor_state.curr_policy_id

                if policy_id not in policy_inputs:
                    policy_inputs[policy_id] = dict(obs=[actor_state.last_obs], rnn_states=[actor_state.rnn_state])
                else:
                    policy_inputs[policy_id]['obs'].append(actor_state.last_obs)
                    policy_inputs[policy_id]['rnn_states'].append(actor_state.rnn_state)
                policy_num_inputs[policy_id] += 1

        for policy_id, policy_input in policy_inputs.items():
            obs_dict = dict()
            observations = policy_input['obs']
            if isinstance(observations[0], (dict, OrderedDict)):
                for key in observations[0].keys():
                    if not isinstance(observations[0][key], str):
                        obs_dict[key] = [o[key] for o in observations]
            else:
                # handle flat observations also as dict
                obs_dict['obs'] = observations

            for key, x in obs_dict.items():
                obs_dict[key] = np.stack(x)

            policy_input['obs'] = obs_dict
            policy_inputs[policy_id] = (policy_num_inputs[policy_id], self.rollout_step, policy_input)

        return policy_inputs

    def _prepare_next_step(self):
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.curr_actions = actor_state.new_rnn_state = None
                actor_state.ready = False

    def reset(self):
        for env_i, e in enumerate(self.envs):
            observations = e.reset()
            for agent_i, obs in enumerate(observations):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.trajectory_add_policy_inputs(obs, actor_state.rnn_state)
                actor_state.last_obs = obs

        policy_inputs = self._format_policy_inputs()
        return policy_inputs

    def advance_rollouts(self, data, policy_version, timing):
        with timing.time_avg('save_policy_outputs'):
            policy_outputs = data['outputs']
            policy_id = data['policy_id']
            all_actors_ready = self._save_policy_outputs(policy_id, policy_outputs, policy_version)
            if not all_actors_ready:
                return None, None, None

        # increment rollout step idx here, before the actual env step because 0-th rollout step comes from env.reset()
        self.rollout_step = (self.rollout_step + 1) % self.cfg.rollout

        complete_rollouts, episodic_stats = [], []

        for env_i, e in enumerate(self.envs):
            with timing.add_time('env_step'):
                actions = [s.curr_actions for s in self.actor_states[env_i]]
                new_obs, rewards, dones, infos = e.step(actions)

            with timing.add_time('overhead'):
                rollouts, stats = self._process_env_step(new_obs, rewards, dones, infos, env_i)
                complete_rollouts.extend(rollouts)
                episodic_stats.extend(stats)

        with timing.add_time('format_inputs'):
            policy_inputs = self._format_policy_inputs()

        self._prepare_next_step()

        return policy_inputs, complete_rollouts, episodic_stats

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
        self, cfg, obs_space, action_space, worker_idx=0,
        task_queue=None, plasma_store_name=None, policy_queues=None, report_queue=None, learner_queues=None,
    ):
        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space

        self.worker_idx = worker_idx
        self.terminate = False

        # random delay in the beginning guarantees that workers will produce complete rollouts more or less
        # uniformly, improving the overall throughput and reducing policy version gap
        self.add_random_delay = not self.cfg.benchmark
        self.rollout_start = None

        self.plasma_store_name = plasma_store_name
        self.is_multiagent = False

        self.vector_size = cfg.num_envs_per_worker
        self.num_splits = cfg.worker_num_splits
        assert self.vector_size >= self.num_splits
        assert self.vector_size % self.num_splits == 0, 'Vector size should be divisible by num_splits'

        # self.env_vectors, self.actor_states, self.episode_rewards = None, None, None
        self.env_runners = None

        self.plasma_client = None
        self.serialization_context = None

        self.serialize_in_background = False
        self.serialization_thread = threading.Thread(target=self._enqueue_loop)
        self.serialization_queue = queue.Queue()

        self.policy_queues = policy_queues
        self.report_queue = report_queue
        self.learner_queues = learner_queues
        self.task_queue = task_queue

        self.process = Process(target=self._run, daemon=True)
        self.process.start()

    def _init(self):
        self.plasma_client = plasma.connect(self.plasma_store_name)
        self.serialization_context = pa.default_serialization_context()
        plasma_store = (self.plasma_client, self.serialization_context)

        log.info('Initializing envs for env runner %d...', self.worker_idx)

        self.env_runners = []
        for split_idx in range(self.num_splits):
            env_runner = VectorEnvRunner(
                self.cfg, self.vector_size // self.num_splits, self.worker_idx, split_idx, plasma_store,
            )
            env_runner.init()
            self.env_runners.append(env_runner)

    def _terminate(self):
        for env_runner in self.env_runners:
            env_runner.close()

        self.terminate = True

    def _enqueue_loop(self):
        plasma_client = plasma.connect(self.plasma_store_name)
        serialization_context = pa.default_serialization_context()

        while True:
            data = safe_get(self.serialization_queue)
            if self.terminate:
                break

            policy_request, data = data
            if policy_request:
                split_idx, policy_inputs = data
                self._enqueue_policy_request(split_idx, policy_inputs, serialization_context, plasma_client)
            else:
                complete_rollouts = data
                self._enqueue_complete_rollouts(complete_rollouts, serialization_context, plasma_client)

    def _enqueue_policy_request(self, split_idx, policy_inputs, serialization_context, plasma_client):
        for policy_id, experience in policy_inputs.items():
            experience = plasma_client.put(
                experience, None, serialization_context=serialization_context,
            )
            policy_request = dict(worker_idx=self.worker_idx, split_idx=split_idx, policy_inputs=experience)
            self.policy_queues[policy_id].put((TaskType.POLICY_STEP, policy_request))

    def _enqueue_complete_rollouts(self, complete_rollouts, serialization_context, plasma_client):
        rollouts_per_policy = dict()

        for rollout in complete_rollouts:
            policy_id = rollout['policy_id']
            if policy_id in rollouts_per_policy:
                rollouts_per_policy[policy_id].append(rollout)
            else:
                rollouts_per_policy[policy_id] = [rollout]

        for policy_id, rollouts in rollouts_per_policy.items():
            rollouts = plasma_client.put(
                rollouts, None, serialization_context=serialization_context,
            )
            self.learner_queues[policy_id].put((TaskType.TRAIN, rollouts))

    def _report_stats(self, stats):
        for report in stats:
            self.report_queue.put(report)

    def _handle_reset(self):
        for split_idx, env_runner in enumerate(self.env_runners):
            policy_inputs = env_runner.reset()

            if self.serialize_in_background:
                self.serialization_queue.put((True, (split_idx, policy_inputs)))
            else:
                self._enqueue_policy_request(split_idx, policy_inputs, self.serialization_context, self.plasma_client)

        log.info('Finished reset for worker %d', self.worker_idx)

    def _advance_rollouts(self, data, timing):
        split_idx = data['split_idx']
        policy_version = data['policy_version']

        if self.rollout_start is None:
            self.rollout_start = time.time()

        policy_inputs, complete_rollouts, episodic_stats = self.env_runners[split_idx].advance_rollouts(
            data, policy_version, timing,
        )

        if episodic_stats:
            self._report_stats(episodic_stats)

        with timing.add_time('enqueue_policy_requests'):
            if policy_inputs is not None:
                if self.serialize_in_background:
                    data = (split_idx, policy_inputs)
                    self.serialization_queue.put((True, data))
                else:
                    self._enqueue_policy_request(
                        split_idx, policy_inputs, self.serialization_context, self.plasma_client,
                    )

        with timing.add_time('complete_rollouts'):
            if complete_rollouts:
                if self.serialize_in_background:
                    self.serialization_queue.put((False, complete_rollouts))
                else:
                    self._enqueue_complete_rollouts(complete_rollouts, self.serialization_context, self.plasma_client)

                if self.add_random_delay:
                    rollout_duration = time.time() - self.rollout_start
                    delay = random.random() * 5 * rollout_duration
                    log.info('Rollout took %.3f sec, sleep for %.3f sec', rollout_duration, delay)
                    time.sleep(delay)
                    self.add_random_delay = False

    def _run(self):
        log.info('Initializing vector env runner %d...', self.worker_idx)

        timing = Timing()
        initialized = False

        self.serialization_thread.start()

        while not self.terminate:
            with timing.add_time('waiting'):
                timeout = 1 if initialized else 1e3
                task_type, data = safe_get(self.task_queue, timeout=timeout)

            if task_type == TaskType.INIT:
                self._init()
                self.task_queue.task_done()
                continue

            if task_type == TaskType.TERMINATE:
                self._terminate()
                self.task_queue.task_done()
                break

            # handling actual workload
            if task_type == TaskType.RESET:
                with timing.add_time('reset'):
                    self._handle_reset()
            elif task_type == TaskType.ROLLOUT_STEP:
                if 'work' not in timing:
                    timing.waiting = 0  # measure waiting only after real work has started

                with timing.add_time('work'):
                    with timing.time_avg('one_step'):
                        self._advance_rollouts(data, timing)

            self.task_queue.task_done()

        if self.worker_idx <= 1:
            log.info('Env runner %d: timing %s', self.worker_idx, timing)

        self.serialization_queue.put(None)
        self.serialization_thread.join()

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.join()

    def request_reset(self):
        self.task_queue.put((TaskType.RESET, None))

    def request_step(self, split, actions):
        data = (split, actions)
        self.task_queue.put((TaskType.ROLLOUT_STEP, data))

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        self.process.join(timeout=2.0)
