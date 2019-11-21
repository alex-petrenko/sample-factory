import copy
import math
import random
import select
import time
from collections import OrderedDict, deque
from enum import Enum
from multiprocessing import JoinableQueue, Process

import numpy as np
import ray
import ray.pyarrow_files.pyarrow as pa
import torch
from ray.pyarrow_files.pyarrow import plasma
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import functional

from algorithms.ppo.agent_ppo import calc_num_elements
from algorithms.utils.action_distributions import calc_num_logits, sample_actions_log_probs, get_action_distribution
from algorithms.utils.algo_utils import num_env_steps, EPS
from algorithms.utils.multi_agent import MultiAgentWrapper
from algorithms.utils.multi_env import safe_get, empty_queue
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


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, cfg):
        super().__init__()

        self.cfg = cfg
        self.action_space = action_space

        def nonlinearity():
            return nn.ELU(inplace=True)

        obs_shape = AttrDict()
        if hasattr(obs_space, 'spaces'):
            for key, space in obs_space.spaces.items():
                obs_shape[key] = space.shape
        else:
            obs_shape.obs = obs_space.shape
        input_ch = obs_shape.obs[0]
        log.debug('Num input channels: %d', input_ch)

        if cfg.encoder == 'convnet_simple':
            conv_filters = [[input_ch, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        elif cfg.encoder == 'minigrid_convnet_tiny':
            conv_filters = [[3, 16, 3, 1], [16, 32, 2, 1], [32, 64, 2, 1]]
        else:
            raise NotImplementedError(f'Unknown encoder {cfg.encoder}')

        conv_layers = []
        for layer in conv_filters:
            if layer == 'maxpool_2x2':
                conv_layers.append(nn.MaxPool2d((2, 2)))
            elif isinstance(layer, (list, tuple)):
                inp_ch, out_ch, filter_size, stride = layer
                conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride))
                conv_layers.append(nonlinearity())
            else:
                raise NotImplementedError(f'Layer {layer} not supported!')

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        log.debug('Convolutional layer output size: %r', self.conv_out_size)

        self.head_out_size = self.conv_out_size

        self.measurements_head = None
        if 'measurements' in obs_shape:
            self.measurements_head = nn.Sequential(
                nn.Linear(obs_shape.measurements[0], 128),
                nonlinearity(),
                nn.Linear(128, 128),
                nonlinearity(),
            )
            measurements_out_size = calc_num_elements(self.measurements_head, obs_shape.measurements)
            self.head_out_size += measurements_out_size

        log.debug('Policy head output size: %r', self.head_out_size)

        self.hidden_size = cfg.hidden_size
        self.linear1 = nn.Linear(self.head_out_size, self.hidden_size)

        fc_output_size = self.hidden_size

        if cfg.use_rnn:
            self.core = nn.GRUCell(fc_output_size, self.hidden_size)
        else:
            self.core = nn.Sequential(
                nn.Linear(fc_output_size, self.hidden_size),
                nonlinearity(),
            )

        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.dist_linear = nn.Linear(self.hidden_size, calc_num_logits(self.action_space))

        self.apply(self.initialize_weights)

        self.train()

    def forward_head(self, obs_dict):
        mean = self.cfg.obs_subtract_mean
        scale = self.cfg.obs_scale

        if abs(mean) > EPS and abs(scale - 1.0) > EPS:
            obs_dict.obs = (obs_dict.obs - mean) * (1.0 / scale)  # convert rgb observations to [-1, 1]

        x = self.conv_head(obs_dict.obs)
        x = x.view(-1, self.conv_out_size)

        if self.measurements_head is not None:
            measurements = self.measurements_head(obs_dict.measurements)
            x = torch.cat((x, measurements), dim=1)

        x = self.linear1(x)
        x = functional.elu(x)  # activation before LSTM/GRU? Should we do it or not?
        return x

    def forward_core(self, head_output, rnn_states, masks):
        if self.cfg.use_rnn:
            x = new_rnn_states = self.core(head_output, rnn_states * masks)
        else:
            x = self.core(head_output)
            new_rnn_states = torch.zeros(x.shape[0])

        return x, new_rnn_states

    def forward_tail(self, core_output):
        values = self.critic_linear(core_output)
        action_logits = self.dist_linear(core_output)
        dist = get_action_distribution(self.action_space, raw_logits=action_logits)

        # for non-trivial action spaces it is faster to do these together
        actions, log_prob_actions = sample_actions_log_probs(dist)

        result = AttrDict(dict(
            actions=actions,
            action_logits=action_logits,
            log_prob_actions=log_prob_actions,
            values=values,
        ))
        return result

    def forward(self, obs_dict, rnn_states, masks=None):
        x = self.forward_head(obs_dict)

        if masks is None:
            masks = torch.ones([x.shape[0], 1]).to(x.device)

        x, new_rnn_states = self.forward_core(x, rnn_states, masks)
        result = self.forward_tail(x)
        result.rnn_states = new_rnn_states
        return result

    @staticmethod
    def initialize_weights(layer):
        if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
            nn.init.orthogonal_(layer.weight.data, gain=1)
            layer.bias.data.fill_(0)
        elif type(layer) == nn.GRUCell:
            nn.init.orthogonal_(layer.weight_ih, gain=1)
            nn.init.orthogonal_(layer.weight_hh, gain=1)
            layer.bias_ih.data.fill_(0)
            layer.bias_hh.data.fill_(0)
        else:
            pass


def make_env_func(cfg, env_config):
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)
    if not hasattr(env, 'num_agents'):
        env = MultiAgentWrapper(env)
    return env


class ActorState:
    """State of a single actor in an environment."""

    def __init__(self, cfg, worker_idx, split_idx, agent_idx):
        self.cfg = cfg
        self.worker_idx = worker_idx
        self.split_idx = split_idx
        self.agent_idx = agent_idx

        self.curr_policy_id = self._sample_random_policy()
        self.rnn_state = self._reset_rnn_state()
        self.last_obs = None

        self.trajectory = dict()
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
        return len(self.trajectory['obs']) if 'obs' in self.trajectory else 0

    def finalize_trajectory(self, done, plasma_client, serialization_context):
        if not done and self.trajectory_len() < self.cfg.rollout:
            return None

        t_id = f'{self.curr_policy_id}_{self.worker_idx}_{self.split_idx}_{self.agent_idx}'
        traj_serialized = plasma_client.put(self.trajectory, None, serialization_context=serialization_context)
        traj_dict = dict(
            t_id=t_id, length=self.trajectory_len(), env_steps=self.env_steps, policy_id=self.curr_policy_id,
            t=traj_serialized,
        )

        self.trajectory = dict()
        self.episode_reward = 0
        self.env_steps = 0
        self.curr_policy_id = self._sample_random_policy()

        return traj_dict

    def update_rnn_state(self, new_rnn_state, done):
        if done:
            self.rnn_state = self._reset_rnn_state()
        else:
            self.rnn_state = new_rnn_state

        return self.rnn_state


class VectorEnvRunner:
    def __init__(self, cfg, num_envs, worker_idx, split_idx, plasma_store_name):
        self.cfg = cfg

        self.num_envs = num_envs
        self.worker_idx = worker_idx
        self.split_idx = split_idx

        self.num_agents = -1  # queried from env

        self.envs, self.actor_states, self.episode_rewards = [], [], []

        self.plasma_client = plasma.connect(plasma_store_name)
        self.serialization_context = pa.default_serialization_context()

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
                actor_state = ActorState(self.cfg, env_i, self.split_idx, agent_idx)
                actor_states_env.append(actor_state)
                episode_rewards_env.append(0.0)

            self.actor_states.append(actor_states_env)
            self.episode_rewards.append(episode_rewards_env)

    def _parse_policy_outputs(self, policy_outputs):
        actions = np.empty((self.num_envs, self.num_agents), dtype=object)
        new_rnn_states = np.empty((self.num_envs, self.num_agents), dtype=object)

        for policy_id, policy_output_obj in policy_outputs.items():
            # deserialized = self.plasma_client.get(
            #     policy_output_obj, -1, serialization_context=self.serialization_context,
            # )
            deserialized = policy_output_obj
            policy_outputs[policy_id] = deserialized

        actors_per_policy = [0 for _ in range(self.cfg.num_policies)]
        for env_i in range(len(self.envs)):
            for agent_i in range(self.num_agents):
                policy_id = self.actor_states[env_i][agent_i].curr_policy_id
                outputs = AttrDict(policy_outputs[policy_id])

                i = actors_per_policy[policy_id]
                actors_per_policy[policy_id] += 1

                actions[env_i][agent_i] = outputs.actions[i]
                new_rnn_states[env_i][agent_i] = outputs.rnn_states[i]

                self.actor_states[env_i][agent_i].trajectory_add_policy_step(
                    outputs.actions[i],
                    outputs.action_logits[i],
                    outputs.log_prob_actions[i],
                    outputs.values[i],
                )

        return actions, new_rnn_states

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
            complete_rollout = actor_state.finalize_trajectory(
                dones[agent_i], self.plasma_client, self.serialization_context,
            )
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
            policy_input_serialized = self.plasma_client.put(
                policy_input, None, serialization_context=self.serialization_context,
            )
            policy_inputs[policy_id] = (policy_num_inputs[policy_id], policy_input_serialized)

        return policy_inputs

    def reset(self):
        for env_i, e in enumerate(self.envs):
            observations = e.reset()
            for agent_i, obs in enumerate(observations):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.trajectory_add_policy_inputs(obs, actor_state.rnn_state)
                actor_state.last_obs = obs

        policy_inputs = self._format_policy_inputs()
        return policy_inputs

    def advance_rollouts(self, policy_outputs, timing):
        with timing.time_avg('parse_policy_outputs'):
            actions, new_rnn_states = self._parse_policy_outputs(policy_outputs)

        complete_rollouts = []
        for env_i, e in enumerate(self.envs):
            with timing.add_time('env_step'):
                new_obs, rewards, dones, infos = e.step(actions[env_i])

            with timing.add_time('overhead'):
                rewards = self._process_rewards(rewards, env_i)
                self._save_env_step(rewards, dones, infos, env_i)

                with timing.add_time('finalize'):
                    complete_rollouts.extend(self._finalize_trajectories(dones, env_i))

                self._save_policy_inputs(new_obs, new_rnn_states[env_i], dones, env_i)

        with timing.add_time('format_output'):
            policy_inputs = self._format_policy_inputs()

        return policy_inputs, complete_rollouts

    def close(self):
        for e in self.envs:
            e.close()


class TaskType(Enum):
    INIT, TERMINATE, RESET, ROLLOUT_STEP, POLICY_STEP = range(5)


class CpuWorker:
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

    def __init__(self, cfg, obs_space, action_space, worker_idx=0, plasma_store_name=None):
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

        self.task_queue, self.result_queue = JoinableQueue(), JoinableQueue()
        self.process = Process(target=self._run, daemon=True)

        self.process.start()

    def _init(self):
        log.info('Initializing envs for env runner %d...', self.worker_idx)

        self.env_runners = []
        for split_idx in range(self.num_splits):
            env_runner = VectorEnvRunner(
                self.cfg, self.vector_size // self.num_splits, self.worker_idx, split_idx, self.plasma_store_name,
            )
            env_runner.init()
            self.env_runners.append(env_runner)

    def _terminate(self):
        for env_runner in self.env_runners:
            env_runner.close()

    def _handle_reset(self):
        policy_inputs = []
        for env_runner in self.env_runners:
            policy_inputs.append(env_runner.reset())

        result = dict(splits=list(range(self.num_splits)), policy_inputs=policy_inputs)
        return result

    def _advance_rollouts(self, data, timing):
        split_idx, policy_outputs = data
        policy_inputs, complete_rollouts = self.env_runners[split_idx].advance_rollouts(policy_outputs, timing)
        return dict(split_idx=split_idx, policy_inputs=policy_inputs, complete_rollouts=complete_rollouts)

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
                    result = self._handle_reset()
            else:
                if 'work' not in timing:
                    timing.waiting = 0  # measure waiting only after real work has started

                with timing.add_time('work'):
                    with timing.time_avg('one_step'):
                        result = self._advance_rollouts(data, timing)

            result['worker_idx'] = self.worker_idx
            result['task_type'] = task_type

            self.result_queue.put(result)
            self.task_queue.task_done()

        if self.worker_idx <= 1:
            log.info('Env runner %d: timing %s', self.worker_idx, timing)

    def await_task(self, task_type, split_idx, data=None):
        """Submit a task and block until it's completed."""

        self.task_queue.put((task_type, split_idx, data))
        self.task_queue.join()

        results = safe_get(self.result_queue)
        self.result_queue.task_done()

        return results

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.join()
        log.info('Env runner %d initialzed...', self.worker_idx)

    def reset(self):
        results = []
        for split in range(self.num_splits):
            _, result = self.await_task(TaskType.RESET, split)
            results.append(result)
        return results

    def request_reset(self):
        self.task_queue.put((TaskType.RESET, None))

    def request_step(self, split, actions):
        data = (split, actions)
        self.task_queue.put((TaskType.ROLLOUT_STEP, data))

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        empty_queue(self.result_queue)
        self.process.join(timeout=2.0)


class GpuWorker:
    def __init__(self, worker_idx, cfg, obs_space, action_space, plasma_store_name):
        log.info('Initializing GPU worker %d', worker_idx)

        self.worker_idx = worker_idx
        self.cfg = cfg

        self.num_tasks = 0

        self.obs_space = obs_space
        self.action_space = action_space

        self.plasma_store_name = plasma_store_name
        self.plasma_client = None
        self.serialization_context = None

        self.device = None
        self.actor_critic = None

        self.num_requests = 0

        self.task_queue, self.result_queue = JoinableQueue(), JoinableQueue()
        self.process = Process(target=self._run, daemon=True)
        self.process.start()

    def _should_log(self):
        log_rate = 50
        return self.num_requests % log_rate == 0

    def _init(self):
        self.result_queue.put(None)
        log.info('GPU worker %d initialized', self.worker_idx)

    def _terminate(self):
        del self.actor_critic
        del self.device

    def _handle_policy_step(self, data, timing):
        policy_id, requests = data

        self.num_requests += 1
        with timing.add_time('policy_step'):
            with timing.add_time('deserialize'):
                observations = AttrDict()
                rnn_states = []
                num_obs_per_actor = []

                for request in requests:
                    actor_idx, split_idx, num_inputs, policy_input = request
                    policy_input = self.plasma_client.get(
                        policy_input, -1, serialization_context=self.serialization_context,
                    )

                    obs = policy_input['obs']
                    for key, x in obs.items():
                        if key in observations:
                            observations[key].append(x)
                        else:
                            observations[key] = [x]

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
                    actor_critic = self.actor_critic[policy_id]
                    policy_outputs = actor_critic(observations, rnn_states)

                with timing.add_time('postprocess'):
                    for key, value in policy_outputs.items():
                        policy_outputs[key] = value.cpu().numpy()

                    output_idx = 0
                    outputs_per_actor = dict()
                    for actor_index, split_idx, num_obs in num_obs_per_actor:
                        outputs = dict()
                        for key, value in policy_outputs.items():
                            outputs[key] = value[output_idx:output_idx + num_obs]

                        with timing.add_time('serialize'):
                            # outputs_per_actor[(actor_index, split_idx)] = self.plasma_client.put(
                            #     outputs, None, serialization_context=self.serialization_context,
                            # )
                            outputs_per_actor[(actor_index, split_idx)] = outputs
                        output_idx += num_obs

            with timing.timeit('result'):
                self.result_queue.put(dict(
                    task_type=TaskType.POLICY_STEP, policy_id=policy_id, outputs_per_actor=outputs_per_actor,
                    gpu_worker_idx=self.worker_idx,
                ))

    def _run(self):
        timing = Timing()

        with timing.timeit('init'):
            self.plasma_client = plasma.connect(self.plasma_store_name)
            self.serialization_context = pa.default_serialization_context()

            # initialize the Torch modules
            self.device = torch.device('cuda')
            self.actor_critic = dict()
            for policy_id in range(self.cfg.num_policies):
                self.actor_critic[policy_id] = ActorCritic(self.obs_space, self.action_space, self.cfg)
                self.actor_critic[policy_id].to(self.device)

        while True:
            with timing.add_time('gpu_waiting'):
                task_type, data = safe_get(self.task_queue)

            if task_type == TaskType.INIT:
                self._init()
            elif task_type == TaskType.TERMINATE:
                self._terminate()
                break
            elif task_type == TaskType.POLICY_STEP:
                if 'work' not in timing:
                    timing.gpu_waiting = 0  # measure waiting time only after real work has started

                with timing.add_time('work'):
                    self._handle_policy_step(data, timing)

        log.info('Gpu worker timing: %s', timing)

    # def train(self, policy_id, training_data):
    #     # TODO: pass the latest parameters too!
    #     # TODO: return updated parameters after training!
    #     return dict(policy_id=policy_id, worker_index=self.worker_index, weights=None)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        return safe_get(self.result_queue)

    def policy_step(self, data):
        self.task_queue.put((TaskType.POLICY_STEP, data))

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        empty_queue(self.result_queue)
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

        p.add_argument('--recurrence', default=32, type=int, help='Trajectory length for backpropagation through time. If recurrence=1 there is no backpropagation through time, and experience is shuffled completely randomly')
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
        p.add_argument('--num_policies', default=8, type=int, help='Number of policies to train jointly')
        p.add_argument('--num_learners', default=2, type=int, help='Number of GPU learners')
        p.add_argument('--macro_batch', default=6144, type=int, help='Amount of experience to collect per policy before passing experience to the learner')

    def __init__(self, cfg):
        super().__init__(cfg)

        self.plasma_store_name = None

        self.obs_space = self.action_space = None

        self.cpu_workers = None
        self.gpu_workers = None
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

    def create_worker(self, idx):
        return CpuWorker(self.cfg, self.obs_space, self.action_space, idx, plasma_store_name=self.plasma_store_name)

    # noinspection PyProtectedMember
    def init_subset(self, indices):
        workers = dict()
        started_reset = dict()
        for i in indices:
            w = self.create_worker(i)
            w.init()
            w.request_reset()
            workers[i] = w
            started_reset[i] = time.time()

        fastest_reset_time = None

        results = dict()
        while True:
            queues = [w.result_queue._reader for w in workers.values()]
            ready, _, _ = select.select(queues, [], [], 0.1)

            for ready_queue in ready:
                w = None
                for worker in workers.values():
                    if ready_queue._handle == worker.result_queue._reader._handle:
                        w = worker
                        break
                assert w is not None

                result = safe_get(w.result_queue)
                result = AttrDict(result)
                results[w.worker_idx] = result
                log.info('Finished reset for worker %d', w.worker_idx)
                if fastest_reset_time is None:
                    fastest_reset_time = time.time() - started_reset[w.worker_idx]
                    log.debug('Fastest reset in %.3f seconds', fastest_reset_time)

            if len(results) >= len(workers):
                break

            for worker_idx, w in workers.items():
                if worker_idx in results:
                    continue
                if fastest_reset_time is None:
                    continue

                time_passed = time.time() - started_reset[w.worker_idx]
                if time_passed > min(fastest_reset_time * 1.5, fastest_reset_time + 10):
                    # if it takes more than 1.5x the usual time to reset, this worker is probably stuck
                    log.error('Worker %d seems to be stuck (%.3f). Reset!', w.worker_idx, time_passed)
                    stuck_worker = w
                    stuck_worker.process.kill()

                    new_worker = self.create_worker(worker_idx)
                    new_worker.init()
                    new_worker.request_reset()
                    started_reset[worker_idx] = time.time()

                    workers[worker_idx] = new_worker
                    del stuck_worker

        return list(workers.values()), results

    # noinspection PyUnresolvedReferences
    def init_workers(self):
        log.info('Initializing GPU workers...')
        self.gpu_workers = []
        for i in range(self.cfg.num_learners):
            gpu_worker = GpuWorker(i, self.cfg, self.obs_space, self.action_space, self.plasma_store_name)
            self.gpu_workers.append(gpu_worker)

        log.info('Initializing actors...')

        self.cpu_workers = []
        reset_results = []
        max_parallel_init = 8
        worker_indices = list(range(self.cfg.num_workers))
        for i in range(0, self.cfg.num_workers, max_parallel_init):
            workers, reset_result = self.init_subset(worker_indices[i:i + max_parallel_init])
            self.cpu_workers.extend(workers)
            reset_results.extend(reset_result.values())

        self.workers_by_handle = dict()
        for w in self.gpu_workers:
            self.workers_by_handle[w.result_queue._reader._handle] = w
        for w in self.cpu_workers:
            self.workers_by_handle[w.result_queue._reader._handle] = w

        # wait for GPU workers to finish initializing
        for w in self.gpu_workers:
            w.init()

        return reset_results

    def start_rollouts(self, reset_results):
        for res in reset_results:
            res = AttrDict(res)
            worker_idx, splits, policy_inputs = res.worker_idx, res.splits, res.policy_inputs
            for policy_inputs, split in zip(policy_inputs, splits):
                rollout_step = dict(
                    split_idx=split, worker_idx=worker_idx, policy_inputs=policy_inputs, complete_rollouts=[],
                )
                self.process_rollout(rollout_step)

    def process_task_result(self, task_result):
        task_type = task_result.task_type
        if task_type == TaskType.ROLLOUT_STEP:
            self.process_rollout(task_result)
        elif task_type == TaskType.POLICY_STEP:
            self.process_policy_step(task_result)
        # elif task_type == APPO.TASK_TRAIN:
        #     self.process_train(task_result)
        else:
            raise Exception(f'Unknown task {task_type}')

    def process_rollout(self, result):
        step_policy_inputs = result['policy_inputs']
        worker_idx = result['worker_idx']
        split_idx = result['split_idx']
        for policy_id, policy_input in step_policy_inputs.items():
            num_inputs, policy_input = policy_input
            self.policy_inputs[policy_id].append((worker_idx, split_idx, num_inputs, policy_input))
            self.policy_outputs[(worker_idx, split_idx)][policy_id] = None  # waiting for outputs to be computed

        for complete_rollout in result['complete_rollouts']:
            env_steps = complete_rollout['env_steps']
            self.num_frames += env_steps
            pass

            # rollout_timing = result['timing']
            # self.last_timing = Timing(rollout_timing)
            #
            # self.save_trajectories(result)
            # self.num_frames += result['num_steps']  # total collected experience

    def process_policy_step(self, result):
        policy_id = result['policy_id']
        outputs_per_actor = result['outputs_per_actor']
        gpu_worker_idx = result['gpu_worker_idx']

        # distribute calculated policy outputs among corresponding CPU actors
        for actor, policy_output_obj in outputs_per_actor.items():
            worker_idx, split_idx = actor
            self.policy_outputs[(worker_idx, split_idx)][policy_id] = policy_output_obj

            if all([o is not None for o in self.policy_outputs[(worker_idx, split_idx)].values()]):
                # finished calculating policy outputs for this actor
                worker = self.cpu_workers[worker_idx]
                policy_outputs = self.policy_outputs[(worker_idx, split_idx)]
                worker.request_step(split_idx, policy_outputs)

                self.policy_outputs[(worker_idx, split_idx)] = dict()  # delete old policy outputs

        self.gpu_workers[gpu_worker_idx].num_tasks -= 1
        assert self.gpu_workers[gpu_worker_idx].num_tasks >= 0

    def process_train(self, train_result):
        train_result = ray.get(train_result)
        policy_id = train_result['policy_id']
        gpu_worker_idx = train_result['worker_idx']

        # TODO: update the latest weights for the policy
        # TODO: increment policy version
        log.info('Finished training for policy %d', policy_id)

        self.gpu_workers[gpu_worker_idx].active_task = None
        self.currently_training.remove(policy_id)

    def save_trajectories(self, rollout):
        rollout_trajectories = rollout['trajectories']
        for t in rollout_trajectories:
            policy_id = t['policy_id']
            if policy_id not in self.trajectories:
                self.trajectories[policy_id] = dict(traj=[], traj_len=[])
            self.trajectories[policy_id]['traj'].append(t['t'])
            self.trajectories[policy_id]['traj_len'].append(t['length'])

    def process_experience(self):
        #TODO
        return

        free_gpu_workers = [w for w in self.gpu_workers if w.active_task is None]

        for policy_id, traj_data in self.trajectories.items():
            length = sum(traj_data['traj_len'])
            if length < self.cfg.macro_batch:
                continue
            if policy_id in self.currently_training:
                continue

            # enough experience for the policy to start training
            if len(free_gpu_workers) < 1:
                # always leave at least one GPU worker for action computation
                break

            gpu_worker = free_gpu_workers.pop()
            self.train(policy_id, gpu_worker)

    def train(self, policy_id, gpu_worker):
        """Train policy `policy_id` on given GPU worker."""
        traj_data = self.trajectories[policy_id]
        total_len = 0
        num_segments = 0
        for t, t_len in zip(traj_data['traj'], traj_data['traj_len']):
            total_len += t_len
            num_segments += 1
            if total_len >= self.cfg.macro_batch:
                break

        training_data = traj_data['traj'][:num_segments]

        # leave remaining data to train later
        self.trajectories[policy_id]['traj'] = traj_data['traj'][num_segments:]
        self.trajectories[policy_id]['traj_len'] = traj_data['traj_len'][num_segments:]

        # TODO: pass latest policy parameters
        train_task = gpu_worker.train.remote(policy_id, training_data)

        assert policy_id not in self.currently_training
        self.currently_training.add(policy_id)
        gpu_worker.active_task = train_task

    def compute_policy_steps(self):
        free_gpu_workers = [w for w in self.gpu_workers if w.num_tasks <= 0]

        while len(free_gpu_workers) > 0:
            gpu_worker = free_gpu_workers.pop()
            task = self.compute_policy_task()
            if task is None:
                break

            gpu_worker.policy_step(task)
            gpu_worker.num_tasks += 1

    def compute_policy_task(self):
        policy_with_most_data = -1
        max_num_inputs = 0

        # find the policy with the most observations collected so far
        for policy_id, policy_inputs in enumerate(self.policy_inputs):
            num_inputs = 0
            for policy_input in policy_inputs:
                _, _, n, _ = policy_input
                num_inputs += n

            if num_inputs > max_num_inputs:
                policy_with_most_data = policy_id
                max_num_inputs = len(policy_inputs)

        if max_num_inputs <= 0:
            # no new experience from the policies
            return None

        selected_policy = policy_with_most_data
        task = (selected_policy, self.policy_inputs[selected_policy])
        self.policy_inputs[selected_policy] = []
        return task

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
        reset_results = self.init_workers()
        self.start_rollouts(reset_results)

        log.info('Collecting experience...')
        timing = Timing()
        queues = [w.result_queue._reader for w in self.cpu_workers]
        queues.extend([w.result_queue._reader for w in self.gpu_workers])

        with timing.timeit('experience'):
            while self.num_frames < 1000000:  # TODO: stopping condition
                ready, _, _ = select.select(queues, [], [], 0.001)

                for ready_queue in ready:
                    w = self.workers_by_handle[ready_queue._handle]
                    result = safe_get(w.result_queue)
                    result = AttrDict(result)
                    self.process_task_result(result)

                self.process_experience()
                self.compute_policy_steps()

                self.print_stats()

        for w in self.cpu_workers:
            w.close()
        for w in self.gpu_workers:
            w.close()
        for w in self.cpu_workers:
            w.join()
        for w in self.gpu_workers:
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
