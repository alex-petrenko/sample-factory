import math
import psutil
import random
import time
from collections import OrderedDict

import gym
import numpy as np
import torch
from gym import spaces, Wrapper
from torch.multiprocessing import Process as TorchProcess, Event

from algorithms.appo.appo_utils import TaskType, set_step_data, cores_for_worker_process
from algorithms.appo.population_based_training import PbtTask
from algorithms.utils.algo_utils import num_env_steps
from algorithms.utils.multi_agent import MultiAgentWrapper
from algorithms.utils.multi_env import safe_get
from envs.create_env import create_env
from envs.doom.multiplayer.doom_multiagent_wrapper import MultiAgentEnv
from utils.timing import Timing
from utils.utils import log, AttrDict, memory_consumption_mb


class DictObservationsWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents
        self.observation_space = gym.spaces.Dict(dict(obs=self.observation_space))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return [dict(obs=o) for o in obs]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return [dict(obs=o) for o in obs], rew, done, info


def make_env_func(cfg, env_config):
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)
    if not hasattr(env, 'num_agents') or env.num_agents <= 1:
        env = MultiAgentWrapper(env)
    if not isinstance(env.observation_space, spaces.Dict):
        env = DictObservationsWrapper(env)
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


# noinspection PyProtectedMember
def update_reward_shaping_scheme(env, agent_idx, reward_shaping):
    if hasattr(env.unwrapped, '_reward_shaping_wrapper'):
        env.unwrapped._reward_shaping_wrapper.reward_shaping_scheme = reward_shaping
    elif isinstance(env.unwrapped, MultiAgentEnv):
        env.unwrapped.set_env_attr(agent_idx, 'unwrapped._reward_shaping_wrapper.reward_shaping_scheme', reward_shaping)


class ActorState:
    """State of a single actor in an environment."""

    def __init__(self, cfg, env, worker_idx, split_idx, env_idx, agent_idx, num_traj_buffers, pbt_reward_shaping):
        self.cfg = cfg
        self.env = env
        self.worker_idx = worker_idx
        self.split_idx = split_idx
        self.env_idx = env_idx
        self.agent_idx = agent_idx

        self.curr_policy_id = self._sample_random_policy()

        # start with zeros in the RNN state
        self.policy_inputs = dict()
        self.set_step_data(self.policy_inputs, 'rnn_states', torch.zeros([self.cfg.hidden_size]))
        self.policy_output_tensors = None  # to be initialized by worker
        self.output_tensor_names = self.output_tensor_sizes = None
        self.policy_outputs = dict()

        self.new_rnn_state = None

        self.ready = False

        self.trajectories = [dict() for _ in range(num_traj_buffers)]

        self.num_trajectories = 0
        self.rollout_env_steps = 0

        self.last_episode_reward = 0
        self.last_episode_duration = 0
        self.last_episode_true_reward = 0

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

    def _reset_rnn_state(self):
        self.policy_inputs['rnn_states'].fill_(0.0)

    def curr_actions(self):
        return self.policy_outputs['actions'].type(torch.int32).numpy()

    @staticmethod
    def set_step_data(dictionary, key, data):
        if isinstance(data, (dict, OrderedDict)):
            # in case of e.g. dictionary observations
            if key not in dictionary:
                dictionary[key] = dict()

            for dict_key, dict_value in data.items():
                set_step_data(dictionary[key], dict_key, dict_value)
        else:
            set_step_data(dictionary, key, data)

    def _trajectory_add_arg_simple(self, dictionary, key, value, rollout_step):
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, (int, float, bool, list, tuple, np.float32)):
            tensor = torch.tensor(value)
        else:
            raise RuntimeError('Only numpy arrays and torch tensors are supported!')

        if key not in dictionary:
            # allocate shared memory tensor of the appropriate shape
            rollout_tensor = torch.stack([tensor] * self.cfg.rollout)
            rollout_tensor.share_memory_()
            dictionary[key] = rollout_tensor
            assert rollout_step == 0
        else:
            # copy data from the numpy array directly into shared memory
            dictionary[key][rollout_step].copy_(tensor)  # is this the fastest way?

        assert dictionary[key].is_shared()

    def _trajectory_add_data(self, trajectory, data, rollout_step, exclude=None):
        for arg_name, arg_value in data.items():
            # skip certain tensors that the learner does not need (for performance)
            if exclude is not None and arg_name in exclude:
                continue

            if isinstance(arg_value, (dict, OrderedDict)):
                if arg_name not in trajectory:
                    trajectory[arg_name] = dict()

                # in case of e.g. dictionary observations
                for dict_key, dict_value in arg_value.items():
                    self._trajectory_add_arg_simple(trajectory[arg_name], dict_key, dict_value, rollout_step)
            else:
                self._trajectory_add_arg_simple(trajectory, arg_name, arg_value, rollout_step)

    def trajectory_add_data(self, data, rollout_step, traj_buffer_idx, exclude=None):
        trajectory = self.trajectories[traj_buffer_idx]
        self._trajectory_add_data(trajectory, data, rollout_step, exclude)

    def record_env_step(self, reward, done, info, rollout_step, traj_buffer_idx):
        # add policy inputs to the trajectory (obs, rnn_states)
        self.trajectory_add_data(self.policy_inputs, rollout_step, traj_buffer_idx)

        # add policy outputs to the trajectory (actions, action probs, etc.)
        # do not add the new rnn state, because it will overwrite the actual policy input
        self.trajectory_add_data(self.policy_outputs, rollout_step, traj_buffer_idx, exclude=['rnn_states'])

        trajectory = self.trajectories[traj_buffer_idx]
        self._trajectory_add_arg_simple(trajectory, 'rewards', reward, rollout_step)
        self._trajectory_add_arg_simple(trajectory, 'dones', done, rollout_step)

        env_steps = num_env_steps([info])
        self.rollout_env_steps += env_steps
        self.last_episode_duration += env_steps

        if done:
            self.new_episode = True
            if self.cfg.with_pbt:
                self.last_episode_true_reward = info.get('true_reward', 0)

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
        else:
            # rnn state output of the current step is input for the next step
            self.policy_inputs['rnn_states'].copy_(self.policy_outputs['rnn_states'])

    def episodic_stats(self):
        stats = dict(reward=self.last_episode_reward, len=self.last_episode_duration)

        if self.cfg.with_pbt:
            stats['true_reward'] = self.last_episode_true_reward

        report = dict(episodic=stats, policy_id=self.curr_policy_id)
        self.last_episode_reward = self.last_episode_duration = self.last_episode_true_reward = 0
        return report


class VectorEnvRunner:
    def __init__(self, cfg, num_envs, worker_idx, split_idx, pbt_reward_shaping):
        self.cfg = cfg

        self.num_envs = num_envs
        self.worker_idx = worker_idx
        self.split_idx = split_idx

        self.num_traj_buffers = -1

        self.rollout_step = 0
        self.traj_buffer_idx = 0  # current shared trajectory buffer to use
        self.traj_buffers_initialized = None

        # whenever the buffer is ready the learner will just flip the bool value in the buffer using shared memory
        self.traj_buffer_ready = None

        self.num_agents = -1  # queried from env

        self.envs, self.actor_states, self.episode_rewards = [], [], []

        self.pbt_reward_shaping = pbt_reward_shaping

    def _on_num_agents_known(self):
        """Some extra initialization once num of agents is queried from the env."""

        # calculate how many buffers are required per env runner to collect one "macro batch" for training
        # once macro batch is collected, all buffers will be released
        # we could have just copied the tensors on the learner to avoid this complicated logic, but it's better for
        # performance to keep data in shared buffers until they're needed
        samples_per_iteration = self.cfg.macro_batch * self.cfg.num_policies
        self.num_traj_buffers = samples_per_iteration / (self.cfg.num_workers * self.cfg.num_envs_per_worker * self.num_agents * self.cfg.rollout)

        # make sure we have enough buffers to actually never wait
        # usually it'll be just two buffers and we swap back and forth
        self.num_traj_buffers *= 3

        # make sure we have at least two to swap between so we never actually have to wait
        self.num_traj_buffers = math.ceil(max(self.num_traj_buffers, 2))

        self.traj_buffers_initialized = [False] * self.num_traj_buffers

        # whenever the buffer is ready the learner will just flip the bool value in the buffer using shared memory
        log.debug(
            'Actor %d-%d allocating %d trajectory buffers', self.worker_idx, self.split_idx, self.num_traj_buffers,
        )
        self.traj_buffer_ready = torch.ones([self.num_envs, self.num_agents, self.num_traj_buffers], dtype=torch.uint8)
        self.traj_buffer_ready.share_memory_()

    def init(self):
        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i
            env_config = AttrDict({'worker_index': self.worker_idx, 'vector_index': vector_idx})
            # log.info('Creating env %r... %d-%d-%d', env_config, self.worker_idx, self.split_idx, env_i)
            env = make_env_func(self.cfg, env_config=env_config)

            if self.num_agents == -1:
                self.num_agents = env.num_agents
                self._on_num_agents_known()

            env.seed(self.worker_idx * 1000 + env_i)
            self.envs.append(env)

            actor_states_env, episode_rewards_env = [], []
            for agent_idx in range(self.num_agents):
                actor_state = ActorState(
                    self.cfg, env, self.worker_idx, self.split_idx, env_i, agent_idx, self.num_traj_buffers,
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
                    # via shared memory mechanism the new data is already copied into the shared tensors
                    policy_outputs = torch.split(
                        actor_state.policy_output_tensors,
                        split_size_or_sections=actor_state.output_tensor_sizes,
                        dim=0,
                    )
                    for tensor_idx, name in enumerate(actor_state.output_tensor_names):
                        actor_state.policy_outputs[name] = policy_outputs[tensor_idx]

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
        rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)
        rewards = rewards * self.cfg.reward_scale
        return rewards

    def _process_env_step(self, new_obs, rewards, dones, infos, env_i):
        episodic_stats = []
        env_actor_states = self.actor_states[env_i]

        rewards = self._process_rewards(rewards, env_i)

        for agent_i in range(self.num_agents):
            actor_state = env_actor_states[agent_i]

            actor_state.record_env_step(
                rewards[agent_i], dones[agent_i], infos[agent_i], self.rollout_step, self.traj_buffer_idx,
            )

            actor_state.set_step_data(actor_state.policy_inputs, 'obs', new_obs[agent_i])
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

        return rollouts

    def _format_policy_request(self):
        policy_request = dict()

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                policy_id = actor_state.curr_policy_id

                data = (env_i, agent_i, self.rollout_step)

                if policy_id not in policy_request:
                    policy_request[policy_id] = []
                policy_request[policy_id].append(data)

        return policy_request

    def _prepare_next_step(self):
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.ready = False

    def policy_input_tensors(self):
        policy_input_tensors = dict()

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                policy_input_tensors[(env_i, agent_i)] = actor_state.policy_inputs

        return policy_input_tensors

    def rollout_tensors(self, traj_buff_idx):
        traj_buffers = dict()
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                traj_buffers[(env_i, agent_i)] = actor_state.trajectories[traj_buff_idx]

        return traj_buffers

    def init_policy_output_tensors(self, env_i, agent_i, tensors, tensor_names, tensor_sizes):
        actor_state = self.actor_states[env_i][agent_i]
        actor_state.policy_output_tensors = tensors
        actor_state.output_tensor_names = tensor_names
        actor_state.output_tensor_sizes = tensor_sizes

    def reset(self):
        for env_i, e in enumerate(self.envs):
            observations = e.reset()
            for agent_i, obs in enumerate(observations):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.set_step_data(actor_state.policy_inputs, 'obs', obs)
                # rnn state is already initialized at zero

            log.debug(
                'Reset progress w:%d-%d finished %d/%d initializing envs...',
                self.worker_idx, self.split_idx, env_i + 1, len(self.envs),
            )

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

        policy_request = self._format_policy_request()
        self._prepare_next_step()

        return policy_request, complete_rollouts, episodic_stats

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
        self, cfg, obs_space, action_space, worker_idx, task_queue, policy_queues, report_queue, learner_queues,
        policy_worker_queues,
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

        self.vector_size = cfg.num_envs_per_worker
        self.num_splits = cfg.worker_num_splits
        assert self.vector_size >= self.num_splits
        assert self.vector_size % self.num_splits == 0, 'Vector size should be divisible by num_splits'

        # self.env_vectors, self.actor_states, self.episode_rewards = None, None, None
        self.env_runners = None

        self.policy_queues = policy_queues
        self.report_queue = report_queue
        self.learner_queues = learner_queues
        self.task_queue = task_queue

        self.policy_workers_queues = policy_worker_queues

        self.reward_shaping = [None for _ in range(self.cfg.num_policies)]

        self.critical_error = Event()
        self.process = TorchProcess(target=self._run, daemon=True)
        self.process.start()

    def _init(self):
        log.info('Initializing envs for env runner %d...', self.worker_idx)

        cpu_count = psutil.cpu_count()
        cores = cores_for_worker_process(self.worker_idx, self.cfg.num_workers, cpu_count)

        if cores is not None:
            psutil.Process().cpu_affinity(cores)

        log.debug('Worker %d uses CPU cores %r', self.worker_idx, psutil.Process().cpu_affinity())

        self.env_runners = []
        for split_idx in range(self.num_splits):
            env_runner = VectorEnvRunner(
                self.cfg, self.vector_size // self.num_splits, self.worker_idx, split_idx, self.reward_shaping,
            )
            env_runner.init()
            self.env_runners.append(env_runner)

    def _terminate(self):
        for env_runner in self.env_runners:
            env_runner.close()

        self.terminate = True

    def _init_policy_input_tensors(self, split_idx, tensors):
        # initialize policy input tensors on all policy workers
        for policy_id in range(self.cfg.num_policies):
            for policy_worker_idx in range(self.cfg.policy_workers_per_policy):
                msg = dict(
                    worker_idx=self.worker_idx, split_idx=split_idx, tensors=tensors,
                    policy_id=policy_id, policy_worker_idx=policy_worker_idx,
                )

                log.debug(
                    'Sending input tensors from %d-%d for policy worker %d-%d...',
                    self.worker_idx, split_idx, policy_id, policy_worker_idx,
                )
                self.policy_workers_queues[policy_id][policy_worker_idx].put((TaskType.INIT_TENSORS, msg))

                # wait to make sure tensors on this worker are indeed initialized
                self.policy_workers_queues[policy_id][policy_worker_idx].join()
                log.debug(
                    'Policy worker %d-%d received the tensors from %d-%d!',
                    policy_id, policy_worker_idx, self.worker_idx, split_idx,
                )

    def _init_policy_output_tensors(self, request_data):
        data = AttrDict(request_data)
        self.env_runners[data.split_idx].init_policy_output_tensors(
            data.env_idx, data.agent_idx, data.tensors, data.tensor_names, data.tensor_sizes,
        )

    def _enqueue_policy_request(self, split_idx, policy_inputs):
        for policy_id, requests in policy_inputs.items():
            policy_request = (self.worker_idx, split_idx, requests)
            self.policy_queues[policy_id].put((TaskType.POLICY_STEP, policy_request))

    def _ensure_traj_buffers_initialized(self, env_runner, split_idx):
        traj_buffer_idx = env_runner.traj_buffer_idx
        if env_runner.traj_buffers_initialized[traj_buffer_idx]:
            return

        for policy_id in range(self.cfg.num_policies):
            request = dict(
                policy_id=policy_id, worker_idx=self.worker_idx, split_idx=split_idx,
                tensors=env_runner.rollout_tensors(traj_buffer_idx),
                is_ready_tensor=env_runner.traj_buffer_ready,
                traj_buffer_idx=traj_buffer_idx,
            )
            log.debug(
                'Sending trajectory buffers from %d-%d to learner %d, idx: %d',
                self.worker_idx, split_idx, policy_id, traj_buffer_idx,
            )
            self.learner_queues[policy_id].put((TaskType.INIT_TENSORS, request))

        self.env_runners[split_idx].traj_buffers_initialized[traj_buffer_idx] = True

    def _enqueue_complete_rollouts(self, split_idx, complete_rollouts, timing):
        """Send complete rollouts from VectorEnv to the learner."""
        if self.cfg.sampler_only:
            return

        env_runner = self.env_runners[split_idx]
        traj_buffer_idx = env_runner.traj_buffer_idx

        # mark current set of trajectory buffers unavailable, until the learner processes them
        env_runner.traj_buffer_ready[:, :, traj_buffer_idx] = 0

        self._ensure_traj_buffers_initialized(env_runner, split_idx)

        rollouts_per_policy = dict()
        for rollout in complete_rollouts:
            policy_id = rollout['policy_id']
            if policy_id not in rollouts_per_policy:
                # provide additional information to the learner, so it is clear which event to raise when
                # the buffer is ready
                rollouts_per_policy[policy_id] = dict(
                    rollouts=[], worker_idx=self.worker_idx,
                    split_idx=split_idx, traj_buffer_idx=traj_buffer_idx,
                )

            rollouts_per_policy[policy_id]['rollouts'].append(rollout)

        # switch to the next available trajectory buffer (usually just double-buffering)
        new_traj_buffer_idx = (traj_buffer_idx + 1) % env_runner.num_traj_buffers
        env_runner.traj_buffer_idx = new_traj_buffer_idx

        for policy_id, rollouts in rollouts_per_policy.items():
            self.learner_queues[policy_id].put((TaskType.TRAIN, rollouts))

        # wait for the next set of buffers to be released, if it's not ready yet
        # this should be a no-op, unless we are collecting experience faster than we can learn from it, in which case
        # this will act as a speed adjusting mechanism
        #
        # If the multi-server version of the algorithm is considered, this mechanism should be dropped
        # in favor of explicit data serialization/sharing across machines
        with timing.add_time('wait_buffers'):
            print_warning = True
            while env_runner.traj_buffer_ready[:, :, new_traj_buffer_idx].min() == 0:
                if print_warning:
                    log.warning(
                        ('Waiting for trajectory buffer %d on actor %d-%d. This means learner is the bottleneck. '
                         'Increase batch size or simplify the computation graph to improve throughput'),
                        new_traj_buffer_idx, self.worker_idx, split_idx,
                    )
                    print_warning = False
                time.sleep(0.005)

    def _report_stats(self, stats):
        for report in stats:
            self.report_queue.put(report)

    def _handle_reset(self):
        for split_idx, env_runner in enumerate(self.env_runners):
            policy_inputs = env_runner.reset()
            self._init_policy_input_tensors(split_idx, env_runner.policy_input_tensors())
            self._enqueue_policy_request(split_idx, policy_inputs)
            time.sleep(random.random() * 0.1)  # helps with Doom issues

        log.info('Finished reset for worker %d', self.worker_idx)

    def _advance_rollouts(self, data, timing):
        split_idx = data['split_idx']

        if self.rollout_start is None:
            self.rollout_start = time.time()

        runner = self.env_runners[split_idx]
        policy_request, complete_rollouts, episodic_stats = runner.advance_rollouts(data, timing)

        if episodic_stats:
            self._report_stats(episodic_stats)

        with timing.add_time('enqueue_policy_requests'):
            if policy_request is not None:
                self._enqueue_policy_request(split_idx, policy_request)

        with timing.add_time('complete_rollouts'):
            if complete_rollouts:
                self._enqueue_complete_rollouts(split_idx, complete_rollouts, timing)

                if self.add_random_delay:
                    rollout_duration = time.time() - self.rollout_start
                    delay = random.random() * 3 * rollout_duration
                    log.info('Rollout took %.3f sec, sleep for %.3f sec', rollout_duration, delay)
                    time.sleep(delay)
                    self.add_random_delay = False

    def _process_pbt_task(self, pbt_task):
        task_type, data = pbt_task

        if task_type == PbtTask.UPDATE_REWARD_SCHEME:
            policy_id, new_reward_shaping_scheme = data
            self.reward_shaping[policy_id] = new_reward_shaping_scheme

    def _run(self):
        log.info('Initializing vector env runner %d...', self.worker_idx)

        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()
        initialized = False

        last_report = time.time()
        with torch.no_grad():
            while not self.terminate:
                with timing.add_time('waiting'), timing.timeit('wait_actor'):
                    timeout = 1 if initialized else 1e3
                    task_type, data = safe_get(self.task_queue, timeout=timeout)

                if task_type == TaskType.INIT:
                    self._init()
                    continue

                if task_type == TaskType.TERMINATE:
                    self._terminate()
                    break

                try:
                    # handling actual workload
                    if task_type == TaskType.RESET:
                        with timing.add_time('reset'):
                            self._handle_reset()
                    elif task_type == TaskType.INIT_TENSORS:
                        self._init_policy_output_tensors(data)
                    elif task_type == TaskType.ROLLOUT_STEP:
                        if 'work' not in timing:
                            timing.waiting = 0  # measure waiting only after real work has started

                        with timing.add_time('work'), timing.timeit('one_step'):
                            self._advance_rollouts(data, timing)
                    elif task_type == TaskType.PBT:
                        self._process_pbt_task(data)

                except RuntimeError as exc:
                    log.warning('Error while processing data w: %d, exception: %s', self.worker_idx, exc)
                    log.warning('Terminate process...')
                    self.terminate = True
                    self.critical_error.set()
                except KeyboardInterrupt:
                    self.terminate = True

                if time.time() - last_report > 5.0 and 'one_step' in timing:
                    timing_stats = dict(wait_actor=timing.wait_actor, step_actor=timing.one_step)
                    memory_mb = memory_consumption_mb()
                    stats = dict(memory_actor=memory_mb)
                    self.report_queue.put(dict(timing=timing_stats, stats=stats))
                    last_report = time.time()

            if self.worker_idx <= 1:
                log.info('Env runner %d: timing %s', self.worker_idx, timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.put((TaskType.EMPTY, None))
        while self.task_queue.qsize() > 0:
            time.sleep(0.01)

    def request_reset(self):
        self.task_queue.put((TaskType.RESET, None))
        self.task_queue.put((TaskType.EMPTY, None))

    def request_step(self, split, actions):
        data = (split, actions)
        self.task_queue.put((TaskType.ROLLOUT_STEP, data))

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        self.process.join(timeout=2.0)
