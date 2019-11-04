import copy
import math
import time
from collections import OrderedDict, deque

import ray
import torch
import numpy as np

from torch import nn
from torch.nn import functional
from tensorboardX import SummaryWriter

from algorithms.ppo.agent_ppo import calc_num_elements
from algorithms.utils.action_distributions import calc_num_logits, sample_actions_log_probs, get_action_distribution
from algorithms.utils.algo_utils import num_env_steps, EPS
from algorithms.utils.multi_agent import MultiAgentWrapper
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


class Agent:
    """Entity responsible for acting in the environment, e.g. collecting rollouts."""

    def __init__(self, cfg):
        self.cfg = cfg


class Learner:
    """Entity responsible for learning from experience."""

    def __init__(self, cfg):
        self.cfg = cfg


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

        if 'obs_mem' in obs_shape:
            self.head_out_size += self.conv_out_size

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

        self.mem_head = None
        if cfg.mem_size > 0:
            mem_out_size = 128
            self.mem_head = nn.Sequential(
                nn.Linear(cfg.mem_size * cfg.mem_feature, mem_out_size),
                nonlinearity(),
            )
            fc_output_size += mem_out_size

        if cfg.use_rnn:
            self.core = nn.GRUCell(fc_output_size, self.hidden_size)
        else:
            self.core = nn.Sequential(
                nn.Linear(fc_output_size, self.hidden_size),
                nonlinearity(),
            )

        if cfg.mem_size > 0:
            self.memory_write = nn.Linear(self.hidden_size, cfg.mem_feature)

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

        if self.cfg.obs_mem:
            obs_mem = self.conv_head(obs_dict.obs_mem)
            obs_mem = obs_mem.view(-1, self.conv_out_size)
            x = torch.cat((x, obs_mem), dim=1)

        if self.measurements_head is not None:
            measurements = self.measurements_head(obs_dict.measurements)
            x = torch.cat((x, measurements), dim=1)

        x = self.linear1(x)
        x = functional.elu(x)  # activation before LSTM/GRU? Should we do it or not?
        return x

    def forward_core(self, head_output, rnn_states, masks, memory):
        if self.mem_head is not None:
            memory = self.mem_head(memory)
            head_output = torch.cat((head_output, memory), dim=1)

        if self.cfg.use_rnn:
            x = new_rnn_states = self.core(head_output, rnn_states * masks)
        else:
            x = self.core(head_output)
            new_rnn_states = torch.zeros(x.shape[0])

        memory_write = None
        if self.cfg.mem_size > 0:
            memory_write = self.memory_write(x)

        return x, new_rnn_states, memory_write

    def forward_tail(self, core_output):
        values = self.critic_linear(core_output)
        action_logits = self.dist_linear(core_output)
        dist = get_action_distribution(self.action_space, raw_logits=action_logits)

        # for complex action spaces it is faster to do these together
        actions, log_prob_actions = sample_actions_log_probs(dist)

        result = AttrDict(dict(
            actions=actions,
            action_logits=action_logits,
            log_prob_actions=log_prob_actions,
            action_distribution=dist,
            values=values,
        ))
        return result

    def forward(self, obs_dict, rnn_states, masks=None):
        x = self.forward_head(obs_dict)

        if masks is None:
            masks = torch.ones([x.shape[0], 1]).to(x.device)

        x, new_rnn_states, memory_write = self.forward_core(x, rnn_states, masks, obs_dict.get('memory', None))
        result = self.forward_tail(x)
        result.rnn_states = new_rnn_states
        result.memory_write = memory_write
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


class AgentAPPO(Agent):
    def __init__(self, cfg, env):
        super().__init__(cfg)

        # TODO
        torch.set_num_threads(1)
        self.device = 'cpu'
        # self.device = torch.device('cuda')

        # initialize the Torch module
        self.actor_critic = ActorCritic(env.observation_space, env.action_space, cfg)
        self.actor_critic.to(self.device)

        self.observations = None
        self.obs_dict = None

        self.rnn_states = None
        self.policy_output = None

        self.trajectories = None

    def preprocess_observations(self, observations):
        """
        Unpack dict observations.
        That is, convert a list of dicts into a dict of lists (numpy arrays).
        """
        if len(observations) <= 0:
            return observations

        obs_dict = AttrDict()
        if isinstance(observations[0], (dict, OrderedDict)):
            for key in observations[0].keys():
                if not isinstance(observations[0][key], str):
                    obs_dict[key] = [o[key] for o in observations]
        else:
            # handle flat observations also as dict
            obs_dict.obs = observations

        for key, x in obs_dict.items():
            obs_dict[key] = torch.from_numpy(np.stack(x)).to(self.device).float()

        self.observations = observations
        self.obs_dict = obs_dict

    def reset_rnn_states(self):
        batch_size = len(self.obs_dict.obs)
        self.rnn_states = torch.zeros(batch_size, self.cfg.hidden_size).to(self.device)

    def update_rnn_states(self, dones):
        if all(dones):
            self.reset_rnn_states()
        else:
            self.rnn_states = self.policy_output.rnn_states

    def step(self):
        if self.rnn_states is None:
            self.reset_rnn_states()

        self.policy_output = self.actor_critic(self.obs_dict, self.rnn_states)
        return self.policy_output

    def _trajectory_add_args(self, args):
        for arg_name, arg_value in args.items():
            if arg_value is None:
                continue

            if isinstance(arg_value, torch.Tensor):
                arg_value = arg_value.cpu().numpy()

            for i, x in enumerate(arg_value):
                if arg_name not in self.trajectories[i]:
                    self.trajectories[i][arg_name] = [x]
                else:
                    self.trajectories[i][arg_name].append(x)

    def _trajectory_add_attributes(
            self, obs, rnn_states, actions, action_logits, log_prob_actions, values, rewards, dones,
    ):
        args = copy.copy(locals())
        del args['self']  # only args passed to the function without "self"
        self._trajectory_add_args(args)

    def update_trajectories(self, rewards, dones):
        if self.trajectories is None:
            self.trajectories = [dict() for _ in range(len(rewards))]

        res = self.policy_output

        self._trajectory_add_attributes(
            self.observations, self.rnn_states,
            res.actions, res.action_logits, res.log_prob_actions, res.values,
            rewards, dones,
        )

    def calculate_next_values(self):
        next_values = self.actor_critic(self.obs_dict, self.rnn_states).values
        next_values = next_values.cpu().numpy()
        for t, v in zip(self.trajectories, next_values):
            t['values'].append(v)


@ray.remote(num_cpus=0.5)
# @ray.remote(num_gpus=0.01)
# @ray.remote
class ActorWorker:
    def __init__(self, worker_index, cfg):
        log.info('Initializing worker %d', worker_index)

        self.cfg = cfg

        self.with_training = True  # TODO: test mode
        self.tmp_actions = None

        # auxiliary fields for the driver process
        self.rollout = None

        def make_env_func(env_config_):
            return create_env(cfg.env, cfg=cfg, env_config=env_config_)

        self.worker_index = worker_index
        env_config = AttrDict({'worker_index': worker_index, 'vector_index': 0})

        self.env = make_env_func(env_config)

        if not hasattr(self.env, 'num_agents'):
            self.env = MultiAgentWrapper(self.env)

        self.policies = dict()
        self.policy_actor_map = dict()
        for policy_id in range(self.cfg.num_policies):
            policy_name = f'policy_{policy_id}'
            self.policies[policy_name] = AgentAPPO(self.cfg, self.env)
            self.policy_actor_map[policy_name] = []

        # number of simultaneous agents in the environment
        self.num_agents = self.env.num_agents

        # default actor-policy map (can be changed during training)
        self.actor_policy_map = []
        default_policy_id = list(self.policies.keys())[0]
        for agent_id in range(self.num_agents):
            self.actor_policy_map.append(default_policy_id)
            self.policy_actor_map[default_policy_id].append(agent_id)

        initial_observations = self.env.reset()
        self.preprocess_observations(initial_observations)

        self.trajectory_counter = 0
        self.episode_rewards = np.zeros(self.num_agents)

    def update_parameters(self, state_dict):
        log.info('Loading new parameters on worker %d', self.worker_index)
        self.agent.actor_critic.load_state_dict(state_dict['model'])

    def preprocess_observations(self, obs):
        """Obs is a list with the size = num_agents."""
        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) <= 0:
                # this policy does not participate in the current rollout
                continue

            policy_obs = []
            for actor_id in actors:
                policy_obs.append(obs[actor_id])

            self.policies[policy_id].preprocess_observations(policy_obs)

    def update_rnn_states(self, dones):
        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) > 0:
                policy_dones = np.asarray(dones)[actors]
                self.policies[policy_id].update_rnn_states(policy_dones)

    def policy_step(self):
        actions = [None] * self.num_agents

        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) <= 0:
                # this policy does not participate in the current rollout
                continue

            policy_outputs = self.policies[policy_id].step()
            actions = policy_outputs.actions.cpu().numpy()

            for i, agent_index in enumerate(actors):
                policy_action = actions[i]
                actions[agent_index] = policy_action

        return actions

    def update_trajectories(self, rewards, dones):
        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) <= 0:
                continue

            policy_rewards = np.asarray(rewards)[actors]
            policy_dones = np.asarray(dones)[actors]
            self.policies[policy_id].update_trajectories(policy_rewards, policy_dones)

    def calculate_next_values(self):
        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) > 0:
                self.policies[policy_id].calculate_next_values()

    def process_rewards(self, rewards):
        rewards = np.asarray(rewards, dtype=np.float32)
        self.episode_rewards += rewards

        rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)
        rewards = rewards * self.cfg.reward_scale
        return rewards

    def process_episode_rewards(self, episode_rewards):
        policy_episode_rewards = dict()
        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) > 0:
                policy_episode_rewards[policy_id] = episode_rewards[actors]

        return policy_episode_rewards

    def finalize_trajectories(self, length, timing):
        trajectories = []

        for policy_id, policy in self.policies.items():
            if policy.trajectories is not None and len(policy.trajectories) > 0:
                for t in policy.trajectories:
                    t_id = f'{self.worker_index}_{self.trajectory_counter}_{policy_id}'
                    with timing.add_time('store_traj'):
                        t_obj_id = ray.put(t)
                    traj_dict = dict(t_id=t_id, length=length, policy_id=policy_id, t=t_obj_id)
                    trajectories.append(traj_dict)
                    self.trajectory_counter += 1

                policy.trajectories = None

        # TODO: calculate GAE?
        return trajectories

    def generate_rollout(self):
        timing = Timing()
        actions = self.tmp_actions

        rollout_step = num_steps = 0

        rollout = dict(worker_index=self.worker_index)

        with timing.timeit('rollout'):
            with torch.no_grad():
                for rollout_step in range(self.cfg.rollout):
                    with timing.add_time('generate_actions'):
                        if self.with_training or actions is None:
                            actions = self.policy_step()
                            self.tmp_actions = actions  # TODO: remove

                    # wait for all the workers to complete an environment step
                    with timing.add_time('env_step'):
                        new_obs, rewards, dones, infos = self.env.step(actions)

                    if self.with_training:
                        with timing.add_time('overhead'):
                            rewards = self.process_rewards(rewards)
                            self.update_trajectories(rewards, dones)
                            self.preprocess_observations(new_obs)
                            self.update_rnn_states(dones)

                    num_steps += num_env_steps(infos)
                    if all(dones):
                        rollout['episode_rewards'] = self.process_episode_rewards(self.episode_rewards)
                        break

                if self.with_training:
                    with timing.timeit('finalize'):
                        self.calculate_next_values()
                        rollout['trajectories'] = self.finalize_trajectories(rollout_step + 1, timing)

        rollout['num_steps'] = num_steps
        rollout['timing'] = dict(timing)

        return rollout

    def close(self):
        self.env.close()


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

        p.add_argument('--num_envs', default=96, type=int, help='Number of environments to collect experience from. Size of the training batch is rollout X num_envs')
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
        p.add_argument('--num_policies', default=8, type=int, help='Number of policies to train jointly')

        # EXPERIMENTAL: external memory
        p.add_argument('--mem_size', default=0, type=int, help='Number of external memory cells')
        p.add_argument('--mem_feature', default=64, type=int, help='Size of the memory cell (dimensionality)')
        p.add_argument('--obs_mem', default=False, type=str2bool, help='Observation-based memory')

    def __init__(self, cfg):
        super().__init__(cfg)

        self.workers = None
        self.trajectories = dict()

    def initialize(self):
        if not ray.is_initialized():
            ray.init(local_mode=False)  # TODO

    def finalize(self):
        ray.shutdown()

    @staticmethod
    def start_rollout(worker):
        rollout = worker.generate_rollout.remote()
        worker.rollout = rollout

    def save_trajectories(self, rollout):
        rollout_trajectories = rollout['trajectories']
        for t in rollout_trajectories:
            policy_id = t['policy_id']
            if policy_id not in self.trajectories:
                self.trajectories[policy_id] = dict(length=0, traj=[])
            self.trajectories[policy_id]['length'] += t['length']
            self.trajectories[policy_id]['traj'].append(t['t'])

    def learn(self):
        self.workers = [
            ActorWorker.remote(i, self.cfg) for i in range(self.cfg.num_workers)
        ]

        for w in self.workers:
            self.start_rollout(w)

        fps_stats = deque([], maxlen=10)
        last_fps_report = time.time()
        num_frames = 0
        fps_stats.append((time.time(), num_frames))
        last_timing = {}

        while True:
            # TODO: stopping condition

            ready, remaining = ray.wait([w.rollout for w in self.workers], num_returns=1, timeout=0.1)

            for rollout in ready:
                rollout = ray.get(rollout)
                worker_index = rollout['worker_index']
                rollout_timing = rollout['timing']
                last_timing = Timing(rollout_timing)

                self.save_trajectories(rollout)

                num_frames += rollout['num_steps']
                self.start_rollout(self.workers[worker_index])

            now = time.time()
            if now - last_fps_report > 1.0:
                past_moment, past_frames = fps_stats[0]
                fps = (num_frames - past_frames) / (now - past_moment)
                log.info('Fps in the last %.1f sec is %.1f', now - past_moment, fps)
                log.debug('Rollout timing %s', last_timing)
                fps_stats.append((now, num_frames))
                last_fps_report = time.time()
