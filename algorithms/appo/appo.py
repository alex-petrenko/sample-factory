import copy
import math
import random
import time
from collections import OrderedDict, deque
from queue import Empty

import numpy as np
import ray
import ray.pyarrow_files.pyarrow as pa
import torch
from ray.pyarrow_files.pyarrow import plasma
from tensorboardX import SummaryWriter
from torch.multiprocessing import JoinableQueue, Process

from algorithms.appo.model import ActorCritic
from algorithms.appo.appo_utils import TaskType, dict_of_lists_append
from algorithms.appo.learner import LearnerWorker
from algorithms.utils.algo_utils import num_env_steps
from algorithms.utils.multi_agent import MultiAgentWrapper
from algorithms.utils.multi_env import safe_get, queue_join_timeout
from envs.create_env import create_env
from utils.timing import Timing
from utils.utils import summaries_dir, experiment_dir, AttrDict, log, str2bool


class Algorithm:
    @classmethod
    def add_cli_args(cls, parser):
        p = parser

        p.add_argument('--seed', default=42, type=int, help='Set a fixed seed value')

        p.add_argument('--initial_save_rate', default=1000, type=int,
                       help='Save model every N steps in the beginning of training')
        p.add_argument('--keep_checkpoints', default=4, type=int, help='Number of model checkpoints to keep')

        p.add_argument('--stats_episodes', default=100, type=int, help='How many episodes to average to measure performance (avg. reward etc)')

        p.add_argument('--learning_rate', default=1e-4, type=float, help='LR')

        p.add_argument('--train_for_steps', default=int(1e10), type=int, help='Stop training after this many SGD steps')
        p.add_argument('--train_for_env_steps', default=int(1e10), type=int, help='Stop training after this many environment steps')
        p.add_argument('--train_for_seconds', default=int(1e10), type=int, help='Stop training after this many seconds')

        # observation preprocessing
        p.add_argument('--obs_subtract_mean', default=0.0, type=float, help='Observation preprocessing, mean value to subtract from observation (e.g. 128.0 for 8-bit RGB)')
        p.add_argument('--obs_scale', default=1.0, type=float, help='Observation preprocessing, divide observation tensors by this scalar (e.g. 128.0 for 8-bit RGB)')

        # RL
        p.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
        p.add_argument(
            '--reward_scale', default=1.0, type=float,
            help=('Multiply all rewards but this factor before feeding into RL algorithm.'
                  'Sometimes the overall scale of rewards is too high which makes value estimation a harder regression task.'
                  'Loss values become too high which requires a smaller learning rate, etc.'),
        )
        p.add_argument('--reward_clip', default=10.0, type=float, help='Clip rewards between [-c, c]. Default [-10, 10] virtually means no clipping for most envs')

        # policy size and configuration
        p.add_argument('--encoder', default='convnet_simple', type=str, help='Type of the policy head (e.g. convolutional encoder)')
        p.add_argument('--hidden_size', default=512, type=int, help='Size of hidden layer in the model, or the size of RNN hidden state in recurrent model (e.g. GRU)')

    def __init__(self, cfg):
        self.cfg = cfg

        # TODO:
        # if self.cfg.seed is not None:
        #     log.info('Settings fixed seed %d', self.cfg.seed)
        #     torch.manual_seed(self.cfg.seed)
        #     np.random.seed(self.cfg.seed)

        self.train_step = self.env_steps = 0

        self.total_train_seconds = 0
        self.last_training_step = time.time()

        self.best_avg_reward = math.nan

        summary_dir = summaries_dir(experiment_dir(cfg=self.cfg))
        self.writer = SummaryWriter(summary_dir, flush_secs=10)


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
        self.last_obs = None

        self.curr_actions = self.new_rnn_state = None
        self.ready = False

        self.trajectory = dict()
        self.num_trajectories = 0
        self.episode_reward = 0
        self.env_steps = 0

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

    def trajectory_add_env_step(self, rewards, dones, info):
        self._trajectory_add_args(rewards=rewards, dones=dones)
        self.env_steps += num_env_steps([info])

    def trajectory_add_policy_step(self, actions, action_logits, log_prob_actions, values):
        args = copy.copy(locals())
        del args['self']  # only args passed to the function without "self"
        self._trajectory_add_args(**args)

    def trajectory_len(self):
        key = 'rewards'  # can be anything
        return len(self.trajectory[key]) if key in self.trajectory else 0

    def finalize_trajectory(self, done):
        if not done and self.trajectory_len() < self.cfg.rollout:
            return None

        obs_dict = transform_dict_observations(self.trajectory['obs'])
        self.trajectory['obs'] = obs_dict

        t_id = f'{self.curr_policy_id}_{self.worker_idx}_{self.split_idx}_{self.env_idx}_{self.agent_idx}_{self.num_trajectories}'
        traj_dict = dict(
            t_id=t_id, length=self.trajectory_len(), env_steps=self.env_steps, policy_id=self.curr_policy_id,
            t=self.trajectory,
        )

        self.trajectory = dict()
        self.episode_reward = 0
        self.env_steps = 0
        self.curr_policy_id = self._sample_random_policy()
        self.num_trajectories += 1

        return traj_dict

    def update_rnn_state(self, new_rnn_state, done):
        if done:
            self.rnn_state = self._reset_rnn_state()
        else:
            self.rnn_state = new_rnn_state

        return self.rnn_state


class VectorEnvRunner:
    def __init__(self, cfg, num_envs, worker_idx, split_idx, plasma_store):
        self.cfg = cfg

        self.num_envs = num_envs
        self.worker_idx = worker_idx
        self.split_idx = split_idx

        self.num_agents = -1  # queried from env

        self.envs, self.actor_states, self.episode_rewards = [], [], []

        self.plasma_client, self.serialization_context = plasma_store

    def init(self):
        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i
            env_config = AttrDict({'worker_index': self.worker_idx, 'vector_index': vector_idx})
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

    def _save_policy_outputs(self, policy_id, policy_outputs):
        policy_outputs = self.plasma_client.get(
                policy_outputs, -1, serialization_context=self.serialization_context,
        )
        all_actors_ready = True

        i = 0
        for env_i in range(len(self.envs)):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                actor_policy = actor_state.curr_policy_id

                if actor_policy == policy_id:
                    outputs = AttrDict(policy_outputs)
                    actor_state.curr_actions = outputs.actions[i]
                    actor_state.new_rnn_state = outputs.rnn_states[i]
                    actor_state.trajectory_add_policy_step(
                        outputs.actions[i],
                        outputs.action_logits[i],
                        outputs.log_prob_actions[i],
                        outputs.values[i],
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
            self.actor_states[env_i][agent_i].episode_reward += r

        rewards = np.asarray(rewards, dtype=np.float32)
        rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)
        rewards = rewards * self.cfg.reward_scale
        return rewards

    def _save_env_step(self, rewards, dones, infos, env_i):
        for agent_i in range(self.num_agents):
            self.actor_states[env_i][agent_i].trajectory_add_env_step(rewards[agent_i], dones[agent_i], infos[agent_i])

    def _save_policy_inputs(self, new_obs, new_rnn_states, dones, env_i):
        for agent_i in range(self.num_agents):
            actor_state = self.actor_states[env_i][agent_i]
            new_rnn_state = actor_state.update_rnn_state(new_rnn_states[agent_i], dones[agent_i])
            actor_state.trajectory_add_policy_inputs(new_obs[agent_i], new_rnn_state)
            actor_state.last_obs = new_obs[agent_i]

    def _finalize_trajectories(self, dones, env_i):
        complete_rollouts = []
        for agent_i in range(self.num_agents):
            actor_state = self.actor_states[env_i][agent_i]
            complete_rollout = actor_state.finalize_trajectory(dones[agent_i])
            if complete_rollout is not None:
                complete_rollouts.append(complete_rollout)

        return complete_rollouts

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
            policy_inputs[policy_id] = (policy_num_inputs[policy_id], policy_input)

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

    def advance_rollouts(self, data, timing):
        with timing.time_avg('save_policy_outputs'):
            policy_outputs = data['outputs']
            policy_id = data['policy_id']
            all_actors_ready = self._save_policy_outputs(policy_id, policy_outputs)
            if not all_actors_ready:
                return None, None

        complete_rollouts = []
        for env_i, e in enumerate(self.envs):
            with timing.add_time('env_step'):
                actions = [s.curr_actions for s in self.actor_states[env_i]]
                new_obs, rewards, dones, infos = e.step(actions)

            with timing.add_time('overhead'):
                rewards = self._process_rewards(rewards, env_i)
                self._save_env_step(rewards, dones, infos, env_i)

                with timing.add_time('finalize'):
                    complete_rollouts.extend(self._finalize_trajectories(dones, env_i))

                new_rnn_states = [s.new_rnn_state for s in self.actor_states[env_i]]
                self._save_policy_inputs(new_obs, new_rnn_states, dones, env_i)

        with timing.add_time('format_output'):
            policy_inputs = self._format_policy_inputs()

        self._prepare_next_step()

        return policy_inputs, complete_rollouts

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
        task_queue=None, plasma_store_name=None, policy_queues=None, learner_queues=None,
    ):
        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space

        self.worker_idx = worker_idx

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

        self.policy_queues = policy_queues
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

    def _enqueue_policy_request(self, split_idx, policy_inputs):
        for policy_id, experience in policy_inputs.items():
            policy_request = dict(worker_idx=self.worker_idx, split_idx=split_idx, policy_inputs=experience)
            policy_request = self.plasma_client.put(
                policy_request, None, serialization_context=self.serialization_context,
            )
            self.policy_queues[policy_id].put((TaskType.POLICY_STEP, policy_request))

    def _enqueue_complete_rollouts(self, complete_rollouts):
        rollouts_per_policy = dict()

        for rollout in complete_rollouts:
            policy_id = rollout['policy_id']
            if policy_id in rollouts_per_policy:
                rollouts_per_policy[policy_id].append(rollout)
            else:
                rollouts_per_policy[policy_id] = [rollout]

        for policy_id, rollouts in rollouts_per_policy.items():
            rollouts = self.plasma_client.put(
                rollouts, None, serialization_context=self.serialization_context,
            )
            self.learner_queues[policy_id].put((TaskType.TRAIN, rollouts))

    def _handle_reset(self):
        for split_idx, env_runner in enumerate(self.env_runners):
            policy_inputs = env_runner.reset()
            self._enqueue_policy_request(split_idx, policy_inputs)
        log.info('Finished reset for worker %d', self.worker_idx)

    def _advance_rollouts(self, data, timing):
        split_idx = data['split_idx']
        policy_inputs, complete_rollouts = self.env_runners[split_idx].advance_rollouts(data, timing)

        if policy_inputs is not None:
            self._enqueue_policy_request(split_idx, policy_inputs)

        if complete_rollouts is not None and len(complete_rollouts) > 0:
            self._enqueue_complete_rollouts(complete_rollouts)
            pass

    def _run(self):
        log.info('Initializing vector env runner %d...', self.worker_idx)

        timing = Timing()
        initialized = False

        while True:
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


class PolicyWorker:
    def __init__(
            self, worker_idx, policy_id, cfg, obs_space, action_space, plasma_store_name, policy_queue, actor_queues,
            weight_queue,
    ):
        log.info('Initializing GPU worker %d for policy %d', worker_idx, policy_id)

        self.worker_idx = worker_idx
        self.policy_id = policy_id
        self.cfg = cfg

        self.obs_space = obs_space
        self.action_space = action_space

        self.plasma_store_name = plasma_store_name
        self.plasma_client = None
        self.serialization_context = None

        self.device = None
        self.actor_critic = None
        self.shared_model = None

        self.task_queue = policy_queue
        self.actor_queues = actor_queues
        self.weight_queue = weight_queue

        self.num_requests = 0

        self.process = Process(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _should_log(self):
        log_rate = 50
        return self.num_requests % log_rate == 0

    def _init(self):
        log.info('GPU worker %d initialized', self.worker_idx)

    def _terminate(self):
        # log.info('GPU worker %d terminated', self.worker_idx)
        pass

    def _handle_policy_step(self, requests, timing):
        if len(requests) <= 0:
            return

        self.num_requests += 1
        with timing.add_time('policy_step'):
            with timing.add_time('deserialize'):
                observations = AttrDict()
                rnn_states = []
                num_obs_per_actor = []

                for request in requests:
                    request = self.plasma_client.get(
                        request, -1, serialization_context=self.serialization_context,
                    )

                    actor_idx = request['worker_idx']
                    split_idx = request['split_idx']
                    num_inputs, policy_input = request['policy_inputs']

                    dict_of_lists_append(observations, policy_input['obs'])
                    rnn_states.append(policy_input['rnn_states'])
                    num_obs_per_actor.append((actor_idx, split_idx, num_inputs))

            with torch.no_grad():
                with timing.add_time('to_device'):
                    for key, x in observations.items():
                        observations[key] = torch.from_numpy(np.concatenate(x)).to(self.device).float()

                    rnn_states = np.concatenate(rnn_states)
                    rnn_states = torch.from_numpy(rnn_states).to(self.device).float()

                # if self._should_log():
                #     log.info(
                #         'Forward pass for policy %d, num observations in a batch %d, GPU worker %d',
                #         policy_id, rnn_states.shape[0], self.worker_idx,
                #     )

                with timing.add_time('forward'):
                    policy_outputs = self.actor_critic(observations, rnn_states)

                with timing.add_time('postprocess'):
                    for key, value in policy_outputs.items():
                        policy_outputs[key] = value.cpu().numpy()

                    output_idx = 0
                    for actor_index, split_idx, num_obs in num_obs_per_actor:
                        outputs = dict()
                        for key, value in policy_outputs.items():
                            outputs[key] = value[output_idx:output_idx + num_obs]

                        with timing.add_time('serialize'):
                            outputs = self.plasma_client.put(
                                outputs, None, serialization_context=self.serialization_context,
                            )

                            advance_rollout_request = dict(
                                split_idx=split_idx, policy_id=self.policy_id, outputs=outputs,
                            )
                            self.actor_queues[actor_index].put((TaskType.ROLLOUT_STEP, advance_rollout_request))

                        output_idx += num_obs

    def _update_weights(self, weight_update, timing):
        if weight_update is None:
            return

        with timing.timeit('weight_update'):
            policy_version, state_dict = weight_update
            self.actor_critic.load_state_dict(state_dict)

        log.info(
            'Updated weights on worker %d, policy_version %d (%.5f)',
            self.worker_idx, policy_version, timing.weight_update,
        )

    def _run(self):
        timing = Timing()

        with timing.timeit('init'):
            self.plasma_client = plasma.connect(self.plasma_store_name)
            self.serialization_context = pa.default_serialization_context()

            # initialize the Torch modules
            log.info('Initializing model on the policy worker %d...', self.worker_idx)

            torch.set_num_threads(1)
            self.device = torch.device('cuda')
            self.actor_critic = ActorCritic(self.obs_space, self.action_space, self.cfg)
            self.actor_critic.to(self.device)

            log.info('Initialized model on the policy worker %d!', self.worker_idx)

        terminate = False
        while not terminate:
            pending_requests = []
            weight_update = None
            work_done = False

            while True:
                try:
                    task_type, data = self.task_queue.get_nowait()
                    if task_type == TaskType.INIT:
                        self._init()
                    elif task_type == TaskType.TERMINATE:
                        self._terminate()
                        terminate = True
                        break
                    elif task_type == TaskType.POLICY_STEP:
                        pending_requests.append(data)

                    self.task_queue.task_done()
                    work_done = True

                except Empty:
                    break

            self._handle_policy_step(pending_requests, timing)

            while True:
                try:
                    task_type, data = self.weight_queue.get_nowait()
                    if task_type == TaskType.UPDATE_WEIGHTS:
                        weight_update = data
                    self.weight_queue.task_done()
                    work_done = True
                except Empty:
                    break

            self._update_weights(weight_update, timing)

            if not work_done:
                with timing.add_time('gpu_waiting'):
                    time.sleep(0.001)

        log.info('Gpu worker timing: %s', timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.join()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        self.process.join(timeout=5)


class APPO(Algorithm):
    """Async PPO."""

    @classmethod
    def add_cli_args(cls, parser):
        p = parser
        super().add_cli_args(p)

        p.add_argument('--adam_eps', default=1e-6, type=float, help='Adam epsilon parameter (1e-8 to 1e-5 seem to reliably work okay, 1e-3 and up does not work)')
        p.add_argument('--adam_beta1', default=0.9, type=float, help='Adam momentum decay coefficient')
        p.add_argument('--adam_beta2', default=0.999, type=float, help='Adam second momentum decay coefficient')

        p.add_argument('--gae_lambda', default=0.95, type=float, help='Generalized Advantage Estimation discounting')

        p.add_argument('--rollout', default=64, type=int, help='Length of the rollout from each environment in timesteps. Size of the training batch is rollout X num_envs')

        p.add_argument('--num_workers', default=16, type=int, help='Number of parallel environment workers. Should be less than num_envs and should divide num_envs')

        p.add_argument('--use_rnn', default=True, type=str2bool, help='Whether to use RNN core in a policy or not')

        p.add_argument('--ppo_clip_ratio', default=1.1, type=float, help='We use unbiased clip(x, e, 1/e) instead of clip(x, 1+e, 1-e) in the paper')
        p.add_argument('--ppo_clip_value', default=0.2, type=float, help='Maximum absolute change in value estimate until it is clipped. Sensitive to value magnitude')
        p.add_argument('--batch_size', default=1024, type=int, help='PPO minibatch size')
        p.add_argument('--ppo_epochs', default=4, type=int, help='Number of training epochs before a new batch of experience is collected')
        p.add_argument('--target_kl', default=0.02, type=float, help='Target distance from behavior policy at the end of training on each experience batch')
        p.add_argument('--early_stopping', default=False, type=str2bool, help='Early stop training on the experience batch when KL-divergence is too high')

        p.add_argument('--normalize_advantage', default=True, type=str2bool, help='Whether to normalize advantages or not (subtract mean and divide by standard deviation)')

        p.add_argument('--max_grad_norm', default=2.0, type=float, help='Max L2 norm of the gradient vector')

        # components of the loss function
        p.add_argument(
            '--prior_loss_coeff', default=0.0005, type=float,
            help=('Coefficient for the exploration component of the loss function. Typically this is entropy maximization, but here we use KL-divergence between our policy and a prior.'
                  'By default prior is a uniform distribution, and this is numerically equivalent to maximizing entropy.'
                  'Alternatively we can use custom prior distributions, e.g. to encode domain knowledge'),
        )
        p.add_argument('--initial_kl_coeff', default=0.0001, type=float, help='Initial value of KL-penalty coefficient. This is adjusted during the training such that policy change stays close to target_kl')
        p.add_argument('--kl_coeff_large', default=0.0, type=float, help='Loss coefficient for the quadratic KL term')
        p.add_argument('--value_loss_coeff', default=0.5, type=float, help='Coefficient for the critic loss')

        # APPO-specific
        p.add_argument('--num_envs_per_worker', default=2, type=int, help='Number of envs on a single CPU actor')
        p.add_argument('--worker_num_splits', default=2, type=int, help='Typically we split a vector of envs into two parts for "double buffered" experience collection')
        p.add_argument('--num_policies', default=1, type=int, help='Number of policies to train jointly')
        p.add_argument('--policy_workers_per_policy', default=1, type=int, help='Number of GPU workers that compute policy forward pass (per policy)')
        p.add_argument('--macro_batch', default=6144, type=int, help='Amount of experience to collect per policy before passing experience to the learner')

    def __init__(self, cfg):
        super().__init__(cfg)

        self.plasma_store_name = None

        self.obs_space = self.action_space = None

        self.actor_workers = None

        self.policy_workers = dict()
        self.policy_queues = dict()

        self.learner_workers = dict()

        self.workers_by_handle = None

        self.trajectories = dict()
        self.currently_training = set()

        self.policy_inputs = [[] for _ in range(self.cfg.num_policies)]
        self.policy_outputs = dict()
        for worker_idx in range(self.cfg.num_workers):
            for split_idx in range(self.cfg.worker_num_splits):
                self.policy_outputs[(worker_idx, split_idx)] = dict()

        self.last_timing = dict()
        self.num_frames = 0
        self.last_fps_report = time.time()

        self.fps_stats = deque([], maxlen=10)
        self.fps_stats.append((time.time(), self.num_frames))

    def initialize(self):
        if not ray.is_initialized():
            ray.init(local_mode=False)

        global_worker = ray.worker.global_worker
        self.plasma_store_name = global_worker.node.plasma_store_socket_name

        tmp_env = make_env_func(self.cfg, env_config=None)
        self.obs_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space
        tmp_env.close()

    def finalize(self):
        ray.shutdown()

    def create_actor_worker(self, idx, actor_queue):
        learner_queues = {p: w.task_queue for p, w in self.learner_workers.items()}

        return ActorWorker(
            self.cfg, self.obs_space, self.action_space, idx, task_queue=actor_queue,
            plasma_store_name=self.plasma_store_name, policy_queues=self.policy_queues,
            learner_queues=learner_queues,
        )

    # noinspection PyProtectedMember
    def init_subset(self, indices, actor_queues):
        workers = dict()
        started_reset = dict()
        for i in indices:
            w = self.create_actor_worker(i, actor_queues[i])
            w.init()
            w.request_reset()
            workers[i] = w
            started_reset[i] = time.time()

        fastest_reset_time = None
        workers_finished = set()

        while len(workers_finished) < len(workers):
            for w in workers.values():
                done = queue_join_timeout(w.task_queue, timeout=0.001)
                if not done:
                    continue

                if len(workers_finished) <= 0:
                    fastest_reset_time = time.time() - started_reset[w.worker_idx]
                    log.debug('Fastest reset in %.3f seconds', fastest_reset_time)

                workers_finished.add(w.worker_idx)

            for worker_idx, w in workers.items():
                if worker_idx in workers_finished:
                    continue
                if fastest_reset_time is None:
                    continue

                time_passed = time.time() - started_reset[w.worker_idx]
                if time_passed > min(fastest_reset_time * 1.5, fastest_reset_time + 10):
                    # if it takes more than 1.5x the usual time to reset, this worker is probably stuck
                    log.error('Worker %d seems to be stuck (%.3f). Reset!', w.worker_idx, time_passed)
                    stuck_worker = w
                    stuck_worker.process.kill()

                    new_worker = self.create_actor_worker(worker_idx, actor_queues[worker_idx])
                    new_worker.init()
                    new_worker.request_reset()
                    started_reset[worker_idx] = time.time()

                    workers[worker_idx] = new_worker
                    del stuck_worker

        return workers.values()

    # noinspection PyUnresolvedReferences
    def init_workers(self):
        actor_queues = [JoinableQueue() for _ in range(self.cfg.num_workers)]

        weight_queues = dict()
        for policy_id in range(self.cfg.num_policies):
            weight_queues[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                weight_queues[policy_id].append(JoinableQueue())

        log.info('Initializing GPU learners...')
        learner_idx = 0
        for policy_id in range(self.cfg.num_policies):
            learner_worker = LearnerWorker(
                learner_idx, policy_id, self.cfg, self.obs_space, self.action_space, self.plasma_store_name,
                weight_queues[policy_id],
            )
            learner_worker.start_process()
            learner_worker.init()

            self.learner_workers[policy_id] = learner_worker
            learner_idx += 1

        log.info('Initializing GPU workers...')
        policy_worker_idx = 0
        for policy_id in range(self.cfg.num_policies):
            self.policy_workers[policy_id] = []

            policy_queue = JoinableQueue()
            self.policy_queues[policy_id] = policy_queue

            for i in range(self.cfg.policy_workers_per_policy):
                policy_worker = PolicyWorker(
                    policy_worker_idx, policy_id, self.cfg, self.obs_space, self.action_space,
                    self.plasma_store_name, policy_queue, actor_queues, weight_queues[policy_id][i],
                )
                self.policy_workers[policy_id].append(policy_worker)
                policy_worker_idx += 1

        log.info('Initializing actors...')

        self.actor_workers = []
        max_parallel_init = 8
        worker_indices = list(range(self.cfg.num_workers))
        for i in range(0, self.cfg.num_workers, max_parallel_init):
            workers = self.init_subset(worker_indices[i:i + max_parallel_init], actor_queues)
            self.actor_workers.extend(workers)

        # wait for GPU workers to finish initializing
        for policy_id, workers in self.policy_workers.items():
            for w in workers:
                w.start_process()
                w.init()

    def print_stats(self):
        now = time.time()
        if now - self.last_fps_report < 1.0:
            return

        past_moment, past_frames = self.fps_stats[0]
        fps = (self.num_frames - past_frames) / (now - past_moment)
        log.debug('Fps in the last %.1f sec is %.1f. Total num frames: %d', now - past_moment, fps, self.num_frames)
        self.fps_stats.append((now, self.num_frames))
        self.last_fps_report = time.time()

    def learn(self):
        self.init_workers()

        log.info('Collecting experience...')

        timing = Timing()
        with timing.timeit('experience'):
            while self.num_frames < 1000000:  # TODO: stopping condition
                for w in self.learner_workers.values():
                    while True:
                        try:
                            report = w.report_queue.get(timeout=0.01)
                            self.num_frames += report['env_steps']
                        except Empty:
                            break

                self.print_stats()

        all_workers = self.actor_workers
        for workers in self.policy_workers.values():
            all_workers.extend(workers)
        all_workers.extend(self.learner_workers.values())

        for w in all_workers:
            w.close()
        for w in all_workers:
            w.join()

        fps = self.num_frames / timing.experience
        log.info('Collected %d, FPS: %.1f', self.num_frames, fps)
        log.info('Timing: %s', timing)

        time.sleep(0.1)
        ray.shutdown()
        log.info('Done!')


# No training
# W20 V20 S2 G2: 26591FPS
# [2019-11-20 19:32:22,965] Gpu worker timing: init: 3.7416, gpu_waiting: 5.6309, deserialize: 5.3061, obs_dict: 0.0868, to_device: 3.9529, forward: 14.0111, serialize: 5.3650, postprocess: 6.7834, policy_step: 31.0437, work: 31.8166
# [2019-11-20 19:32:22,993] Env runner 0: timing waiting: 0.5965, reset: 20.5919, parse_policy_outputs: 0.0004, env_step: 26.4536, finalize: 3.9813, overhead: 4.7497, format_output: 4.6372, one_step: 0.0234, work: 36.8783

# W20 V20 S1 G2: 24996FPS
# [2019-11-20 19:49:01,397] Gpu worker timing: init: 3.6439, gpu_waiting: 9.9744, deserialize: 3.5391, obs_dict: 0.0786, to_device: 4.1121, forward: 16.6075, serialize: 2.7663, postprocess: 4.0433, policy_step: 29.2388, work: 29.9234
# [2019-11-20 19:49:01,404] Env runner 1: timing waiting: 6.4043, reset: 21.3081, parse_policy_outputs: 0.0006, env_step: 24.1964, finalize: 3.8485, overhead: 4.5882, format_output: 4.0478, one_step: 0.0533, work: 33.5044

# W32 V20 S2 (2 GPU workers): 30370FPS
# [2019-11-20 19:17:19,969] Gpu worker timing: init: 3.7086, gpu_waiting: 3.6520, work: 29.1827
# [2019-11-20 19:17:19,970] Env runner 1: timing waiting: 4.4399, reset: 21.1310, parse_policy_outputs: 0.0007, env_step: 19.1307, finalize: 3.5949, overhead: 4.1450, format_output: 3.8386, one_step: 0.0311, work: 28.1974

# W32 V40 S2 (2 GPU workers): 30701FPS
# [2019-11-20 19:24:17,261] Env runner 0: timing waiting: 1.4417, reset: 42.2417, parse_policy_outputs: 0.0015, env_step: 21.1332, finalize: 3.9994, overhead: 4.6047, format_output: 4.0152, one_step: 0.0813, work: 30.7172
# [2019-11-20 19:24:17,339] Env runner 1: timing waiting: 1.3387, reset: 39.7958, parse_policy_outputs: 0.0026, env_step: 21.2498, finalize: 3.7511, overhead: 4.4223, format_output: 4.2317, one_step: 0.0676, work: 30.8883

# W32 V40 S1 G2: 30529FPS
# [2019-11-20 19:56:44,631] Gpu worker timing: init: 3.5720, gpu_waiting: 8.4949, deserialize: 4.6235, obs_dict: 0.0809, to_device: 4.2894, forward: 9.3965, serialize: 3.6527, postprocess: 4.4345, policy_step: 23.5292, work: 24.2091
# [2019-11-20 19:56:44,669] Env runner 0: timing waiting: 4.6958, reset: 44.1553, parse_policy_outputs: 0.0010, env_step: 19.5480, finalize: 3.8980, overhead: 4.5100, format_output: 3.4880, one_step: 0.1341, work: 28.0031
