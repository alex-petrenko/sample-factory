import copy
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from algorithms.memento.mem_wrapper import MemWrapper, split_env_and_memory_actions
from algorithms.memento.obs_mem_wrapper import ObsMemWrapper
from algorithms.utils.action_distributions import calc_num_logits, get_action_distribution, sample_actions_log_probs
from algorithms.utils.agent import TrainStatus, Agent
from algorithms.utils.algo_utils import calculate_gae, num_env_steps, EPS
from algorithms.utils.multi_env import MultiEnv
from utils.timing import Timing
from utils.utils import log, AttrDict, str2bool


class ExperienceBuffer:
    def __init__(self):
        self.obs = self.actions = self.log_prob_actions = self.rewards = self.dones = self.values = None
        self.action_logits = None
        self.masks = self.rnn_states = None
        self.advantages = self.returns = None

    def reset(self):
        self.obs, self.actions, self.log_prob_actions, self.rewards, self.dones, self.values = [], [], [], [], [], []
        self.action_logits = []
        self.masks, self.rnn_states = [], []
        self.advantages, self.returns = [], []

    def _add_args(self, args):
        for arg_name, arg_value in args.items():
            if arg_name in self.__dict__ and arg_value is not None:
                self.__dict__[arg_name].append(arg_value)

    def add(self, obs, actions, action_logits, log_prob_actions, values, masks, rnn_states, rewards, dones):
        """Argument names should match names of corresponding buffers."""
        args = copy.copy(locals())
        self._add_args(args)

    def _to_tensors(self, device):
        for item, x in self.__dict__.items():
            if x is None:
                continue

            if isinstance(x, list) and isinstance(x[0], torch.Tensor):
                self.__dict__[item] = torch.stack(x)
            elif isinstance(x, list) and isinstance(x[0], dict):
                # e.g. dict observations
                tensor_dict = AttrDict()
                for key in x[0].keys():
                    key_list = [x_elem[key] for x_elem in x]
                    tensor_dict[key] = torch.stack(key_list)
                self.__dict__[item] = tensor_dict
            elif isinstance(x, np.ndarray):
                self.__dict__[item] = torch.tensor(x, device=device)

    def _transform_tensors(self):
        """
        Transform tensors to the desired shape for training.
        Before this function all tensors have shape [T, E, D] where:
            T: time dimension (environment rollout)
            E: number of parallel environments
            D: dimensionality of the individual tensor

        This function will convert all tensors to [E, T, D] and then to [E x T, D], which will allow us
        to split the data into trajectories from the same episode for RNN training.
        """

        def _do_transform(tensor):
            assert len(tensor.shape) >= 2
            return tensor.transpose(0, 1).reshape(-1, *tensor.shape[2:])

        for item, x in self.__dict__.items():
            if x is None:
                continue

            if isinstance(x, dict):
                for key, x_elem in x.items():
                    x[key] = _do_transform(x_elem)
            else:
                self.__dict__[item] = _do_transform(x)

    # noinspection PyTypeChecker
    def finalize_batch(self, gamma, gae_lambda, normalize_advantage):
        device = self.values[0].device

        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.dones = np.asarray(self.dones)

        values = torch.stack(self.values).squeeze(dim=2).cpu().numpy()

        # calculate discounted returns and GAE
        self.advantages, self.returns = calculate_gae(self.rewards, self.dones, values, gamma, gae_lambda)

        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        adv_max, adv_min = self.advantages.max(), self.advantages.min()
        adv_max_abs = max(adv_max, abs(adv_min))
        log.info(
            'Adv mean %.3f std %.3f, min %.3f, max %.3f, max abs %.3f',
            adv_mean, adv_std, adv_min, adv_max, adv_max_abs,
        )

        # normalize advantages if needed
        if normalize_advantage:
            self.advantages = (self.advantages - adv_mean) / max(1e-2, adv_std)

        # values vector has one extra last value that we don't need
        self.values = self.values[:-1]

        # convert lists and numpy arrays to PyTorch tensors
        self._to_tensors(device)
        self._transform_tensors()

        # some scalars need to be converted from [E x T] to [E x T, 1] for loss calculations
        self.returns = torch.unsqueeze(self.returns, dim=1)

    def get_minibatch(self, idx):
        mb = AttrDict()

        for item, x in self.__dict__.items():
            if x is None:
                continue

            if isinstance(x, dict):
                mb[item] = AttrDict()
                for key, x_elem in x.items():
                    mb[item][key] = x_elem[idx]
            else:
                mb[item] = x[idx]

        return mb

    def __len__(self):
        return len(self.actions)


def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements


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

    def forward(self, obs_dict, rnn_states, masks):
        x = self.forward_head(obs_dict)
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


class AgentPPO(Agent):
    """Agent based on PPO algorithm."""

    @classmethod
    def add_cli_args(cls, parser):
        p = parser
        super().add_cli_args(p)

        p.add_argument('--adam_eps', default=1e-6, type=float, help='Adam epsilon parameter (1e-8 to 1e-5 seem to reliably work okay, 1e-3 and up does not work)')

        p.add_argument('--gae_lambda', default=0.95, type=float, help='Generalized Advantage Estimation discounting')

        p.add_argument('--rollout', default=64, type=int, help='Length of the rollout from each environment in timesteps. Size of the training batch is rollout X num_envs')

        p.add_argument('--num_envs', default=96, type=int, help='Number of environments to collect experience from. Size of the training batch is rollout X num_envs')
        p.add_argument('--num_workers', default=16, type=int, help='Number of parallel environment workers. Should be less than num_envs and should divide num_envs')

        p.add_argument('--recurrence', default=32, type=int, help='Trajectory length for backpropagation through time. If recurrence=1 there is no backpropagation through time, and experience is shuffled completely randomly')
        p.add_argument('--use_rnn', default=True, type=str2bool, help='Whether to use RNN core in a policy or not')

        p.add_argument('--ppo_clip_ratio', default=1.1, type=float, help='We use unbiased clip(x, e, 1/e) instead of clip(x, 1+e, 1-e) in the paper')
        p.add_argument('--ppo_clip_value', default=0.2, type=float, help='Maximum absolute change in value estimate until it is clipped. Sensitive to value magnitude')
        p.add_argument('--batch_size', default=1024, type=int, help='PPO minibatch size')
        p.add_argument('--ppo_epochs', default=1, type=int, help='Number of training epochs before a new batch of experience is collected')
        p.add_argument('--target_kl', default=0.02, type=float, help='Target distance from behavior policy at the end of training on each experience batch')

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
        p.add_argument('--value_loss_coeff', default=0.5, type=float, help='Coefficient for the critic loss')

        # EXPERIMENTAL: modified PPO objectives
        p.add_argument('--new_clip', default=False, type=str2bool, help='Apply clipping to min(p, 1-p)')
        p.add_argument('--leaky_ppo', default=0.0, type=float, help='Leaky clipped objective instead of constant. Default: standard PPO objective')

        # EXPERIMENTAL: external memory
        p.add_argument('--mem_size', default=0, type=int, help='Number of external memory cells')
        p.add_argument('--mem_feature', default=64, type=int, help='Size of the memory cell (dimensionality)')
        p.add_argument('--obs_mem', default=False, type=str2bool, help='Observation-based memory')

        # EXPERIMENTAL: trying to stabilize the distribution of hidden states
        p.add_argument('--rnn_dist_loss_coeff', default=0.0, type=float, help='Penalty for the difference in hidden state values, compared to the behavioral policy')

        # EXPERIMENTAL: hidden state clipping
        p.add_argument('--clip_hidden_states', default=math.inf, type=float, help='Clip absolute change in hidden state dimensions. Default (inf) means do not clip')
        p.add_argument('--hidden_states_clip_global', default=True, type=str2bool, help='Clip the absolute global change in hidden state vector, rather than individual components')

    def __init__(self, make_env_func, cfg):
        super().__init__(cfg)

        def make_env(env_config):
            env_ = make_env_func(env_config)

            if cfg.obs_mem:
                env_ = ObsMemWrapper(env_)

            if cfg.mem_size > 0:
                env_ = MemWrapper(env_, cfg.mem_size, cfg.mem_feature)

            return env_

        self.make_env_func = make_env
        env = self.make_env_func(None)  # we need the env to query observation shape, number of actions, etc.

        self.actor_critic = ActorCritic(env.observation_space, env.action_space, cfg)
        self.actor_critic.to(self.device)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), cfg.learning_rate, eps=cfg.adam_eps)

        self.memory = np.zeros([cfg.num_envs, cfg.mem_size, cfg.mem_feature], dtype=np.float32)

        self.kl_coeff = self.cfg.initial_kl_coeff
        self.last_batch_kl_divergence = 0.0
        self.last_batch_value_delta = self.last_batch_value_delta_max = 0.0
        self.last_batch_fraction_clipped = 0.0
        self.last_batch_rnn_dist = 0.0
        self.last_batch_hidden_clipped = 0.0

        env.close()

    def _load_state(self, checkpoint_dict):
        super()._load_state(checkpoint_dict)

        self.kl_coeff = checkpoint_dict['kl_coeff']
        self.actor_critic.load_state_dict(checkpoint_dict['model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])

    def _get_checkpoint_dict(self):
        checkpoint = super()._get_checkpoint_dict()
        checkpoint.update({
            'kl_coeff': self.kl_coeff,
            'model': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        })
        return checkpoint

    def _preprocess_observations(self, observations):
        if len(observations) <= 0:
            return observations

        obs_dict = AttrDict()
        if isinstance(observations[0], dict):
            for key in observations[0].keys():
                if not isinstance(observations[0][key], str):
                    obs_dict[key] = [o[key] for o in observations]
        else:
            # handle flat observations also as dict
            obs_dict.obs = observations

        # add memory
        if self.cfg.mem_size > 0:
            obs_dict.memory = self.memory.copy()
            obs_dict.memory = obs_dict.memory.reshape((self.cfg.num_envs, self.cfg.mem_size * self.cfg.mem_feature))

        for key, x in obs_dict.items():
            obs_dict[key] = torch.from_numpy(np.stack(x)).to(self.device).float()

        mean = self.cfg.obs_subtract_mean
        scale = self.cfg.obs_scale

        if abs(mean) > EPS and abs(scale - 1.0) > EPS:
            obs_dict.obs = (obs_dict.obs - mean) * (1.0 / scale)  # convert rgb observations to [-1, 1]
            if self.cfg.obs_mem:
                obs_dict.obs_mem = (obs_dict.obs_mem - mean) * (1.0 / scale)  # convert rgb observations to [-1, 1]

        return obs_dict

    @staticmethod
    def _preprocess_actions(actor_critic_output):
        actions = actor_critic_output.actions.cpu().numpy()
        return actions

    def _update_memory(self, actions, memory_write, dones):
        if memory_write is None:
            assert self.cfg.mem_size == 0
            return

        memory_write = memory_write.cpu().numpy()

        for env_i, action in enumerate(actions):
            if dones[env_i]:
                self.memory[env_i][:][:] = 0.0
                continue

            _, memory_action = split_env_and_memory_actions(action, self.cfg.mem_size)

            for cell_i, memory_cell_action in enumerate(memory_action):
                if memory_cell_action == 0:
                    # noop action - leave memory intact
                    continue
                else:
                    # write action, update memory cell value
                    self.memory[env_i][cell_i] = memory_write[env_i]

    # noinspection PyUnusedLocal
    def best_action(self, observations, dones=None, rnn_states=None, **kwargs):
        with torch.no_grad():
            observations = self._preprocess_observations(observations)
            masks = self._get_masks(dones)

            if rnn_states is None:
                num_envs = len(dones)
                rnn_states = torch.zeros(num_envs, self.cfg.hidden_size).to(self.device)

            res = self.actor_critic(observations, rnn_states, masks)
            actions = self._preprocess_actions(res)
            return actions, res.rnn_states, res

    # noinspection PyTypeChecker
    def _get_masks(self, dones):
        masks = 1.0 - torch.tensor(dones, device=self.device)
        masks = torch.unsqueeze(masks, dim=1)
        return masks.float()

    def _minibatch_indices(self, experience_size, shuffle=True):
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert experience_size % self.cfg.batch_size == 0

        # indices that will start the mini-trajectories from the same episode (for bptt)
        indices = np.arange(0, experience_size, self.cfg.recurrence)

        if shuffle:
            indices = np.random.permutation(indices)

        # complete indices of mini trajectories, e.g. with recurrence==4: [4, 16] -> [4, 5, 6, 7, 16, 17, 18, 19]
        indices = [np.arange(i, i + self.cfg.recurrence) for i in indices]
        indices = np.concatenate(indices)

        assert len(indices) == experience_size

        num_minibatches = experience_size // self.cfg.batch_size
        minibatches = np.split(indices, num_minibatches)
        return minibatches

    def _policy_loss(self, action_distribution, mb, clip_ratio):
        log_prob_actions = action_distribution.log_prob(mb.actions)
        ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

        # ratio_mean = torch.abs(1.0 - ratio).mean()
        # ratio_min = ratio.min()
        # ratio_max = ratio.max()

        p_old = torch.exp(mb.log_prob_actions)

        if self.cfg.new_clip:
            positive_clip = torch.min(p_old * clip_ratio, 1.0 - (1.0 - p_old) / clip_ratio)
            positive_clip_ratio = torch.exp(torch.log(positive_clip) - mb.log_prob_actions)

            negative_clip = torch.max(p_old / clip_ratio, 1.0 - (1.0 - p_old) * clip_ratio)
            negative_clip_ratio = torch.exp(torch.log(negative_clip) - mb.log_prob_actions)
        else:
            positive_clip_ratio = clip_ratio
            negative_clip_ratio = 1.0 / clip_ratio

        is_adv_positive = (mb.advantages > 0.0).float()
        is_ratio_too_big = (ratio > positive_clip_ratio).float() * is_adv_positive

        is_adv_negative = (mb.advantages < 0.0).float()
        is_ratio_too_small = (ratio < negative_clip_ratio).float() * is_adv_negative

        clipping = is_adv_positive * positive_clip_ratio + is_adv_negative * negative_clip_ratio

        is_ratio_clipped = is_ratio_too_big + is_ratio_too_small
        is_ratio_not_clipped = 1.0 - is_ratio_clipped

        # total_non_clipped = torch.sum(is_ratio_not_clipped).float()
        fraction_clipped = is_ratio_clipped.mean()

        objective = ratio * mb.advantages
        leak = self.cfg.leaky_ppo
        objective_clipped = -leak * ratio * mb.advantages + clipping * mb.advantages * (1.0 + leak)

        policy_loss = -(objective * is_ratio_not_clipped + objective_clipped * is_ratio_clipped).mean()

        return policy_loss, ratio, fraction_clipped

    def _value_loss(self, new_values, mb, clip_value):
        value_clipped = mb.values + torch.clamp(new_values - mb.values, -clip_value, clip_value)
        value_original_loss = (new_values - mb.returns).pow(2)
        value_clipped_loss = (value_clipped - mb.returns).pow(2)
        value_loss = torch.max(value_original_loss, value_clipped_loss).mean()
        value_loss *= self.cfg.value_loss_coeff
        value_delta = torch.abs(new_values - mb.values).mean()
        value_delta_max = torch.abs(new_values - mb.values).max()

        return value_loss, value_delta, value_delta_max

    def _clip_hidden_states(self, new_hidden_states, old_hidden_states, epoch, timestep):
        clip = self.cfg.clip_hidden_states
        if math.isinf(clip):
            # no clipping
            return new_hidden_states, 0.0

        delta = new_hidden_states - old_hidden_states

        if self.cfg.hidden_states_clip_global:
            delta_norm = delta.pow(2)
            delta_norm = torch.sum(delta_norm, dim=-1)
            delta_norm = torch.sqrt(delta_norm)

            clipped = (delta_norm > clip).float()
            not_clipped = 1.0 - clipped
            mean_num_clipped = clipped.mean()

            scale = clip / (delta_norm + EPS)
            scale = scale.unsqueeze(dim=-1)

            clipped = clipped.unsqueeze(dim=-1)
            not_clipped = not_clipped.unsqueeze(dim=-1)

            result = clipped * (scale * delta + old_hidden_states) + not_clipped * new_hidden_states
        else:
            clipped = (delta.abs() > clip).float()
            not_clipped = 1.0 - clipped

            mean_num_clipped = clipped.mean()

            clipped_delta = torch.clamp(delta, -clip, clip)
            clipped_states = old_hidden_states + clipped_delta
            result = clipped * clipped_states + not_clipped * new_hidden_states

        return result, mean_num_clipped

    # noinspection PyUnresolvedReferences
    def _train(self, buffer):
        clip_ratio = self.cfg.ppo_clip_ratio
        clip_value = self.cfg.ppo_clip_value
        recurrence = self.cfg.recurrence

        kl_old = 0.0
        value_delta = value_delta_max = 0.0
        fraction_clipped = 0.0
        rnn_dist = 0.0
        hidden_clipped = 0.0

        old_actor_critic = copy.deepcopy(self.actor_critic)
        new_rnn_states = np.zeros_like(buffer.rnn_states.detach().cpu().numpy())

        for epoch in range(self.cfg.ppo_epochs):
            for batch_num, indices in enumerate(self._minibatch_indices(len(buffer))):
                mb_stats = AttrDict(dict(rnn_dist=0))
                with_summaries = self._should_write_summaries(self.train_step)

                # current minibatch consisting of short trajectory segments with length == recurrence
                mb = buffer.get_minibatch(indices)

                # calculate policy head outside of recurrent loop
                head_outputs = self.actor_critic.forward_head(mb.obs)

                # indices corresponding to 1st frames of trajectory segments
                traj_indices = indices[::self.cfg.recurrence]

                # initial rnn states
                rnn_states = buffer.rnn_states[traj_indices]

                # initial memory values
                memory = None
                if self.cfg.mem_size > 0:
                    memory = buffer.obs.memory[traj_indices]

                core_outputs = []

                rnn_dist = 0.0

                for i in range(recurrence):
                    # indices of head outputs corresponding to the current timestep
                    timestep_indices = np.arange(i, self.cfg.batch_size, self.cfg.recurrence)

                    # EXPERIMENTAL: additional loss for difference in hidden states
                    if self.cfg.rnn_dist_loss_coeff > EPS:
                        dist = (rnn_states - mb.rnn_states[timestep_indices]).pow(2)
                        dist = torch.sum(dist, dim=1)
                        dist = dist.mean()
                        rnn_dist += dist

                    rnn_states, hidden_clipped = self._clip_hidden_states(
                        rnn_states, mb.rnn_states[timestep_indices], epoch, i,
                    )
                    new_rnn_states[indices[timestep_indices]] = rnn_states.detach().cpu().numpy()

                    step_head_outputs = head_outputs[timestep_indices]
                    masks = mb.masks[timestep_indices]

                    core_output, rnn_states, memory_write = self.actor_critic.forward_core(
                        step_head_outputs, rnn_states, masks, memory,
                    )
                    core_outputs.append(core_output)

                    behavior_policy_actions = mb.actions[timestep_indices]
                    dones = mb.dones[timestep_indices]

                    if self.cfg.mem_size > 0:
                        # EXPERIMENTAL: external memory
                        mem_actions = behavior_policy_actions[:, -self.cfg.mem_size:]
                        mem_actions = torch.unsqueeze(mem_actions, dim=-1)
                        mem_actions = mem_actions.float()

                        memory_cells = memory.reshape((memory.shape[0], self.cfg.mem_size, self.cfg.mem_feature))

                        write_output = memory_write.repeat(1, self.cfg.mem_size)
                        write_output = write_output.reshape(memory_cells.shape)

                        # noinspection PyTypeChecker
                        new_memories = (1.0 - mem_actions) * memory_cells + mem_actions * write_output
                        memory = new_memories.reshape(memory.shape[0], self.cfg.mem_size * self.cfg.mem_feature)

                        zero_if_done = torch.unsqueeze(1.0 - dones.float(), dim=-1)
                        memory = memory * zero_if_done

                # transform core outputs from [T, Batch, D] to [Batch, T, D] and then to [Batch x T, D]
                # which is the same shape as the minibatch
                core_outputs = torch.stack(core_outputs)
                core_outputs = core_outputs.transpose(0, 1).reshape(-1, *core_outputs.shape[2:])
                assert core_outputs.shape[0] == head_outputs.shape[0]

                # calculate policy tail outside of recurrent loop
                result = self.actor_critic.forward_tail(core_outputs)

                action_distribution = result.action_distribution
                # if batch_num == 0 and epoch == 0:
                #     action_distribution.dbg_print()

                policy_loss, ratio, fraction_clipped = self._policy_loss(action_distribution, mb, clip_ratio)
                value_loss, value_delta, value_delta_max = self._value_loss(result.values, mb, clip_value)

                entropy = action_distribution.entropy().mean()
                kl_prior = action_distribution.kl_prior().mean()

                prior_loss = self.cfg.prior_loss_coeff * kl_prior

                old_action_distribution = get_action_distribution(self.actor_critic.action_space, mb.action_logits)
                kl_old = action_distribution.kl_divergence(old_action_distribution).mean()

                # kl_old_max = action_distribution.kl_divergence(old_action_distribution).max()
                # kl_reverse = old_action_distribution.kl_divergence(action_distribution).mean()
                # log.debug('KL-divergence from old policy distribution is %f (max %f, reverse %f), value delta: %f (max %f)', kl_old, kl_old_max, kl_reverse, value_delta, value_delta_max)
                # log.debug(
                #     'Policy Loss: %.6f, PPO ratio mean %.3f, min %.3f, max %.3f, fraction clipped: %.5f, total_not_clipped %.2f', policy_loss, ratio_mean, ratio_min, ratio_max, fraction_clipped, total_non_clipped,
                # )

                kl_penalty = self.kl_coeff * kl_old

                rnn_dist /= recurrence
                dist_loss = self.cfg.rnn_dist_loss_coeff * rnn_dist

                loss = policy_loss + value_loss + prior_loss + kl_penalty + dist_loss

                if with_summaries:
                    mb_stats.loss = loss
                    mb_stats.value = result.values.mean()
                    mb_stats.entropy = entropy
                    mb_stats.kl_prior = kl_prior
                    mb_stats.value_loss = value_loss
                    mb_stats.prior_loss = prior_loss
                    mb_stats.dist_loss = dist_loss
                    mb_stats.kl_coeff = self.kl_coeff
                    mb_stats.max_abs_logprob = torch.abs(mb.action_logits).max()

                    # we want this statistic for the last batch of the last epoch
                    mb_stats.last_batch_fraction_clipped = self.last_batch_fraction_clipped
                    mb_stats.last_batch_kl_old = self.last_batch_kl_divergence
                    mb_stats.last_batch_value_delta = self.last_batch_value_delta
                    mb_stats.last_batch_value_delta_max = self.last_batch_value_delta_max
                    mb_stats.last_batch_rnn_dist = self.last_batch_rnn_dist
                    mb_stats.last_batch_hidden_clipped = self.last_batch_hidden_clipped

                if epoch == 0 and batch_num == 0 and self.train_step < 1000:
                    # we've done no training steps yet, so all ratios should be equal to 1.0 exactly
                    assert all(abs(r - 1.0) < 1e-4 for r in ratio.detach().cpu().numpy())

                # TODO!!! Figure out whether we need to do it or not
                # Update memories for next epoch
                # if self.acmodel.recurrent and i < self.recurrence - 1:
                #     exps.memory[inds + i + 1] = memory.detach()

                # update the weights
                self.optimizer.zero_grad()
                loss.backward()

                # max_grad = max(
                #     p.grad.max()
                #     for p in self.actor_critic.parameters()
                #     if p.grad is not None
                # )
                # log.debug('max grad back: %.6f', max_grad)

                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                self._after_optimizer_step()

                # collect and report summaries
                if with_summaries:
                    grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in self.actor_critic.parameters()
                        if p.grad is not None
                    ) ** 0.5
                    mb_stats.grad_norm = grad_norm

                    self._report_train_summaries(mb_stats)

        # adjust KL-penalty coefficient if KL divergence at the end of training is high
        if kl_old > self.cfg.target_kl:
            self.kl_coeff *= 1.5
        elif kl_old < self.cfg.target_kl / 2:
            self.kl_coeff /= 1.5
        self.kl_coeff = max(self.kl_coeff, 1e-6)

        self.last_batch_kl_divergence = kl_old
        self.last_batch_value_delta = value_delta
        self.last_batch_value_delta_max = value_delta_max
        self.last_batch_fraction_clipped = fraction_clipped
        self.last_batch_rnn_dist = rnn_dist
        self.last_batch_hidden_clipped = hidden_clipped

        # EXPERIMENTAL - measure avg policy divergence under old and new hidden state distribution
        self._calculate_policy_divergence(buffer, old_actor_critic, buffer.rnn_states.detach().cpu().numpy())
        self._calculate_policy_divergence(buffer, old_actor_critic, new_rnn_states)
        self._calculate_policy_divergence(buffer, self.actor_critic, buffer.rnn_states.detach().cpu().numpy())
        self._calculate_policy_divergence(buffer, self.actor_critic, new_rnn_states)

    def _calculate_policy_divergence(self, buffer, actor_critic, rnn_states):
        """EXPERIMENTAL."""
        clip_ratio = self.cfg.ppo_clip_ratio
        recurrence = self.cfg.recurrence

        indices = self._minibatch_indices(len(buffer), shuffle=False)[0]
        mb = buffer.get_minibatch(indices)
        head_outputs = actor_critic.forward_head(mb.obs)

        core_outputs = []
        for i in range(recurrence):
            # indices of head outputs corresponding to the current timestep
            timestep_indices = np.arange(i, self.cfg.batch_size, self.cfg.recurrence)

            # calculate policy head outside of recurrent loop
            step_head_outputs = head_outputs[timestep_indices]
            masks = mb.masks[timestep_indices]
            curr_rnn_states = torch.from_numpy(rnn_states[indices[timestep_indices]]).to(self.device)

            core_output, _, _ = actor_critic.forward_core(
                step_head_outputs, curr_rnn_states, masks, None,
            )
            core_outputs.append(core_output)

        # transform core outputs from [T, Batch, D] to [Batch, T, D] and then to [Batch x T, D]
        # which is the same shape as the minibatch
        core_outputs = torch.stack(core_outputs)
        core_outputs = core_outputs.transpose(0, 1).reshape(-1, *core_outputs.shape[2:])
        assert core_outputs.shape[0] == head_outputs.shape[0]

        # calculate policy tail outside of recurrent loop
        result = actor_critic.forward_tail(core_outputs)
        action_distribution = result.action_distribution

        policy_loss, ratio, fraction_clipped = self._policy_loss(action_distribution, mb, clip_ratio)

        old_action_distribution = get_action_distribution(actor_critic.action_space, mb.action_logits)
        kl_old = action_distribution.kl_divergence(old_action_distribution).mean()

        log.info('fraction_clipped: %.3f, kl_old: %.6f, min_ratio: %.3f, max_ratio: %.3f', fraction_clipped.item(), kl_old.item(), ratio.min().item(), ratio.max().item())

    def _learn_loop(self, multi_env):
        """Main training loop."""
        buffer = ExperienceBuffer()

        observations = multi_env.reset()
        observations = self._preprocess_observations(observations)

        # actions, rewards and masks do not require backprop so can be stored in buffers
        dones = [True] * self.cfg.num_envs

        rnn_states = torch.zeros(self.cfg.num_envs)
        if self.cfg.use_rnn:
            rnn_states = torch.zeros(self.cfg.num_envs, self.cfg.hidden_size).to(self.device)

        while not self._should_end_training():
            timing = Timing()
            num_steps = 0
            batch_start = time.time()

            buffer.reset()

            # collecting experience
            with torch.no_grad():
                with timing.timeit('experience'):
                    for rollout_step in range(self.cfg.rollout):
                        masks = self._get_masks(dones)
                        res = self.actor_critic(observations, rnn_states, masks)
                        actions = self._preprocess_actions(res)

                        # wait for all the workers to complete an environment step
                        with timing.add_time('env_step'):
                            new_obs, rewards, dones, infos = multi_env.step(actions)
                            rewards = np.asarray(rewards, dtype=np.float32) * self.cfg.reward_scale
                            rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)

                        self._update_memory(actions, res.memory_write, dones)

                        buffer.add(
                            observations,
                            res.actions, res.action_logits, res.log_prob_actions,
                            res.values,
                            masks, rnn_states,
                            rewards, dones,
                        )

                        with timing.add_time('obs'):
                            observations = self._preprocess_observations(new_obs)
                        rnn_states = res.rnn_states

                        num_steps += num_env_steps(infos)

                    # last step values are required for TD-return calculation
                    next_values = self.actor_critic(observations, rnn_states, self._get_masks(dones)).values
                    buffer.values.append(next_values)

                    self.env_steps += num_steps

                with timing.timeit('finalize'):
                    # calculate discounted returns and GAE
                    buffer.finalize_batch(self.cfg.gamma, self.cfg.gae_lambda, self.cfg.normalize_advantage)

            # exit no_grad context, update actor and critic
            with timing.timeit('train'):
                self._train(buffer)

            avg_reward = multi_env.calc_avg_rewards(n=self.cfg.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.cfg.stats_episodes)
            fps = num_steps / (time.time() - batch_start)

            self._maybe_print(avg_reward, avg_length, fps, timing)
            self._maybe_update_avg_reward(avg_reward, multi_env.stats_num_episodes())
            self._report_basic_summaries(fps, avg_reward, avg_length)

        self._on_finished_training()

    def learn(self):
        status = TrainStatus.SUCCESS
        multi_env = None
        try:
            multi_env = MultiEnv(
                self.cfg.num_envs,
                self.cfg.num_workers,
                make_env_func=self.make_env_func,
                stats_episodes=self.cfg.stats_episodes,
            )

            self._learn_loop(multi_env)
        except (Exception, KeyboardInterrupt, SystemExit):
            log.exception('Interrupt...')
            status = TrainStatus.FAILURE
        finally:
            log.info('Closing env...')
            if multi_env is not None:
                multi_env.close()

        return status
