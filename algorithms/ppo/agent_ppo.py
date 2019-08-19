import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from algorithms.memento.mem_wrapper import MemWrapper
from algorithms.utils.action_distributions import calc_num_logits, get_action_distribution, sample_actions_log_probs
from algorithms.utils.agent import TrainStatus, Agent
from algorithms.utils.algo_utils import calculate_gae, num_env_steps, EPS
from algorithms.utils.multi_env import MultiEnv
from utils.timing import Timing
from utils.utils import log, AttrDict


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
    def finalize_batch(self, gamma, gae_lambda):
        device = self.values[0].device

        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.dones = np.asarray(self.dones)

        values = torch.stack(self.values).squeeze(dim=2).cpu().numpy()

        # calculate discounted returns and GAE
        self.advantages, self.returns = calculate_gae(self.rewards, self.dones, values, gamma, gae_lambda)

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

        if cfg.encoder == 'convnet_simple':
            conv_filters = [[3, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
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

        obs_shape = AttrDict()
        if hasattr(obs_space, 'spaces'):
            for key, space in obs_space.spaces.items():
                obs_shape[key] = space.shape
        else:
            obs_shape.obs = obs_space.shape

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

        self.mem_head = None
        if cfg.mem_size > 0:
            mem_out_size = 256
            self.mem_head = nn.Sequential(
                nn.Linear(cfg.mem_size * cfg.mem_feature, mem_out_size),
                nonlinearity(),
            )
            self.head_out_size += mem_out_size

        log.debug('Policy head output size: %r', self.head_out_size)

        self.hidden_size = cfg.hidden_size

        self.linear1 = nn.Linear(self.head_out_size, self.hidden_size)

        if cfg.recurrence == 1:
            # no recurrence
            self.core = None
        else:
            self.core = nn.GRUCell(self.hidden_size, self.hidden_size)

        if cfg.mem_size > 0:
            self.compressed_state = nn.Identity()
            # TODO: compress
            assert cfg.mem_feature == self.hidden_size

            # self.compressed_state = nn.Sequential(
            #     nn.Linear(self.hidden_size, params.mem_feature),
            #     nonlinearity(),
            # )
            # self.reconstructed_state = nn.Linear(params.mem_feature, self.hidden_size)

        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.dist_linear = nn.Linear(self.hidden_size, calc_num_logits(self.action_space))

        self.apply(self.initialize_weights)
        self.apply_gain()

        self.train()

    def apply_gain(self):
        # TODO: do we need this??
        # relu_gain = nn.init.calculate_gain('relu')
        # for i in range(len(self.conv_head)):
        #     if isinstance(self.conv_head[i], nn.Conv2d):
        #         self.conv_head[i].weight.data.mul_(relu_gain)
        #
        # self.linear1.weight.data.mul_(relu_gain)
        pass

    def forward_head(self, obs_dict):
        x = self.conv_head(obs_dict.obs)
        x = x.view(-1, self.conv_out_size)

        if self.measurements_head is not None:
            measurements = self.measurements_head(obs_dict.measurements)
            x = torch.cat((x, measurements), dim=1)

        if self.mem_head is not None:
            mem = self.mem_head(obs_dict.mem)
            x = torch.cat((x, mem), dim=1)

        x = self.linear1(x)
        x = functional.elu(x)  # activation before LSTM/GRU? Should we do it or not?
        return x

    def forward_core(self, head_output, rnn_states, masks):
        if self.cfg.recurrence == 1:
            x = head_output
            new_rnn_states = torch.zeros(x.shape[0])
        else:
            x = new_rnn_states = self.core(head_output, rnn_states * masks)

        compressed_state = reconstructed_state = None
        if self.cfg.mem_size > 0:
            compressed_state = self.compressed_state(x)
            # reconstructed_state = self.reconstructed_state(compressed_state)

        # TODO: loss
        return x, new_rnn_states, compressed_state, reconstructed_state

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
        x, new_rnn_states, compressed_state, reconstructed_state = self.forward_core(x, rnn_states, masks)
        result = self.forward_tail(x)
        result.rnn_states = new_rnn_states
        result.compressed_state = compressed_state
        result.reconstructed_state = reconstructed_state
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

        p.add_argument('--gae_lambda', default=0.95, type=float, help='Generalized Advantage Estimation discounting')

        p.add_argument('--rollout', default=64, type=int, help='Length of the rollout from each environment in timesteps. Size of the training batch is rollout X num_envs')

        p.add_argument('--num_envs', default=96, type=int, help='Number of environments to collect experience from. Size of the training batch is rollout X num_envs')
        p.add_argument('--num_workers', default=16, type=int, help='Number of parallel environment workers. Should be less than num_envs and should divide num_envs')

        p.add_argument('--recurrence', default=16, type=int, help='Trajectory lenght for backpropagation through time. If recurrence=1 the feed-forward model is used (no BPTT)')

        p.add_argument('--ppo_clip_ratio', default=1.1, type=float, help='We use clip(x, e, 1/e) instead of clip(x, 1+e, 1-e) in the paper')
        p.add_argument('--ppo_clip_value', default=0.1, type=float, help='Maximum absolute change in value estimate until it is clipped. Sensitive to value magnitude')
        p.add_argument('--batch_size', default=512, type=int, help='PPO minibatch size')
        p.add_argument('--ppo_epochs', default=4, type=int, help='Number of training epochs before a new batch of experience is collected')

        p.add_argument('--value_loss_coeff', default=0.5, type=float, help='Coefficient for the critic loss')
        p.add_argument('--entropy_loss_coeff', default=0.005, type=float, help='Entropy coefficient')
        p.add_argument('--rnn_dist_loss_coeff', default=0.0, type=float, help='Penalty for the difference in hidden state values, compared to the behavioral policy')

        p.add_argument('--max_grad_norm', default=2.0, type=float, help='Max L2 norm of the gradient vector')

        p.add_argument('--mem_size', default=0, type=int, help='Number of external memory cells')
        p.add_argument('--mem_feature', default=64, type=int, help='Size of the memory cell (dimensionality)')

    def __init__(self, make_env_func, cfg):
        super().__init__(cfg)

        def make_env(env_config):
            env_ = make_env_func(env_config)
            if cfg.mem_size > 0:
                env_ = MemWrapper(env_, cfg.mem_size, cfg.mem_feature)
            return env_

        self.make_env_func = make_env
        env = self.make_env_func(None)  # we need the env to query observation shape, number of actions, etc.

        self.actor_critic = ActorCritic(env.observation_space, env.action_space, self.cfg)
        self.actor_critic.to(self.device)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), cfg.learning_rate)

        env.close()

    def _load_state(self, checkpoint_dict):
        super()._load_state(checkpoint_dict)

        self.actor_critic.load_state_dict(checkpoint_dict['model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])

    def _get_checkpoint_dict(self):
        checkpoint = super()._get_checkpoint_dict()
        checkpoint.update({
            'cfg': self.cfg,
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

        for key, x in obs_dict.items():
            obs_dict[key] = torch.from_numpy(np.stack(x)).to(self.device).float()

        mean = self.cfg.obs_subtract_mean
        scale = self.cfg.obs_scale

        if abs(mean) > EPS and abs(scale - 1.0) > EPS:
            obs_dict.obs = (obs_dict.obs - mean) * (1.0 / scale)  # convert rgb observations to [-1, 1]
        return obs_dict

    def _preprocess_actions(self, actor_critic_output):
        actions = actor_critic_output.actions.cpu().numpy()

        if self.cfg.mem_size > 0:
            actions = np.concatenate([actions, actor_critic_output.compressed_state.cpu().numpy()], axis=1)

        return actions

    def best_action(self, observations, dones=None, rnn_states=None, deterministic=False):
        with torch.no_grad():
            observations = self._preprocess_observations(observations)
            masks = self._get_masks(dones)

            if rnn_states is None:
                num_envs = len(dones)
                rnn_states = torch.zeros(num_envs, self.cfg.hidden_size).to(self.device)

            res = self.actor_critic(observations, rnn_states, masks)
            actions = self._preprocess_actions(res)

            # if deterministic:
            #     raise NotImplementedError('Not supported for some action distributions (TODO!)')
            #     # _, actions = res.action_distribution.probs.max(1)
            # else:
            #     actions = res.action_distribution.sample()

            return actions, res.rnn_states

    # noinspection PyTypeChecker
    def _get_masks(self, dones):
        masks = 1.0 - torch.tensor(dones, device=self.device)
        masks = torch.unsqueeze(masks, dim=1)
        return masks.float()

    def _minibatch_indices(self, experience_size):
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert experience_size % self.cfg.batch_size == 0

        # indices that will start the mini-trajectories from the same episode (for bptt)
        indices = np.arange(0, experience_size, self.cfg.recurrence)
        indices = np.random.permutation(indices)

        # complete indices of mini trajectories, e.g. with recurrence==4: [4, 16] -> [4, 5, 6, 7, 16, 17, 18, 19]
        indices = [np.arange(i, i + self.cfg.recurrence) for i in indices]
        indices = np.concatenate(indices)

        assert len(indices) == experience_size

        num_minibatches = experience_size // self.cfg.batch_size
        minibatches = np.split(indices, num_minibatches)
        return minibatches

    # noinspection PyUnresolvedReferences
    def _train(self, buffer):
        clip_ratio = self.cfg.ppo_clip_ratio
        clip_value = self.cfg.ppo_clip_value
        recurrence = self.cfg.recurrence

        for epoch in range(self.cfg.ppo_epochs):
            for batch_num, indices in enumerate(self._minibatch_indices(len(buffer))):
                mb_stats = AttrDict(dict(
                    value=0, entropy=0, value_loss=0, entropy_loss=0, rnn_dist=0, dist_loss=0,
                ))
                with_summaries = self._should_write_summaries(self.train_step)
                mb_loss = 0.0

                # current minibatch consisting of short trajectory segments with length == recurrence
                mb = buffer.get_minibatch(indices)

                # calculate policy head outside of recurrent loop
                head_outputs = self.actor_critic.forward_head(mb.obs)

                # indices corresponding to 1st frames of trajectory segments
                traj_indices = indices[::self.cfg.recurrence]

                # initial rnn states
                rnn_states = buffer.rnn_states[traj_indices]

                core_outputs = []

                dist_loss = 0.0

                for i in range(recurrence):
                    # indices of head outputs corresponding to the current timestep
                    timestep_indices = np.arange(i, self.cfg.batch_size, self.cfg.recurrence)

                    if self.cfg.rnn_dist_loss_coeff > EPS:
                        dist = (rnn_states - mb.rnn_states[timestep_indices]).pow(2)
                        dist = torch.sum(dist, dim=1)
                        dist = torch.sqrt(dist + EPS)
                        dist = dist.mean()
                        mb_stats.rnn_dist += dist
                        dist_loss += self.cfg.rnn_dist_loss_coeff * dist

                    step_head_outputs = head_outputs[timestep_indices]
                    masks = mb.masks[timestep_indices]

                    # TODO: compressed_state loss
                    core_output, rnn_states, _, _ = self.actor_critic.forward_core(
                        step_head_outputs, rnn_states, masks,
                    )
                    core_outputs.append(core_output)

                # transform core outputs from [T, Batch, D] to [Batch, T, D] and then to [Batch x T, D]
                # which is the same shape as the minibatch
                core_outputs = torch.stack(core_outputs)
                core_outputs = core_outputs.transpose(0, 1).reshape(-1, *core_outputs.shape[2:])
                assert core_outputs.shape[0] == head_outputs.shape[0]

                # calculate policy tail outside of recurrent loop
                result = self.actor_critic.forward_tail(core_outputs)

                action_distribution = result.action_distribution

                ratio = torch.exp(action_distribution.log_prob(mb.actions) - mb.log_prob_actions)  # pi / pi_old
                clipped_advantages = torch.clamp(ratio, 1.0 / clip_ratio, clip_ratio) * mb.advantages
                policy_loss = -torch.min(ratio * mb.advantages, clipped_advantages).mean()

                value_clipped = mb.values + torch.clamp(result.values - mb.values, -clip_value, clip_value)
                value_original_loss = (result.values - mb.returns).pow(2)
                value_clipped_loss = (value_clipped - mb.returns).pow(2)
                value_loss = torch.max(value_original_loss, value_clipped_loss).mean()
                value_loss *= self.cfg.value_loss_coeff

                entropy = action_distribution.entropy().mean()
                entropy_loss = self.cfg.entropy_loss_coeff * -entropy

                dist_loss /= recurrence

                loss = policy_loss + value_loss + entropy_loss + dist_loss
                mb_loss += loss

                if with_summaries:
                    mb_stats.value = result.values.mean()
                    mb_stats.entropy = entropy
                    mb_stats.value_loss = value_loss
                    mb_stats.entropy_loss = entropy_loss
                    mb_stats.dist_loss = dist_loss
                    mb_stats.rnn_dist /= recurrence

                if epoch == 0 and batch_num == 0 and self.train_step < 1000:
                    # we've done no training steps yet, so all ratios should be equal to 1.0 exactly
                    assert all(abs(r - 1.0) < 1e-4 for r in ratio.detach().cpu().numpy())

                # TODO!!! Figure out whether we need to do it or not
                # Update memories for next epoch
                # if self.acmodel.recurrent and i < self.recurrence - 1:
                #     exps.memory[inds + i + 1] = memory.detach()

                # update the weights
                self.optimizer.zero_grad()
                mb_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                self._after_optimizer_step()

                # collect and report summaries
                if with_summaries:
                    mb_stats.loss = mb_loss

                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.actor_critic.parameters()) ** 0.5
                    mb_stats.grad_norm = grad_norm

                    self._report_train_summaries(mb_stats)

    def _learn_loop(self, multi_env):
        """Main training loop."""
        buffer = ExperienceBuffer()

        observations = multi_env.reset()
        observations = self._preprocess_observations(observations)

        # actions, rewards and masks do not require backprop so can be stored in buffers
        dones = [True] * self.cfg.num_envs

        rnn_states = torch.zeros(self.cfg.num_envs)
        if self.cfg.recurrence > 1:
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
                        new_obs, rewards, dones, infos = multi_env.step(actions)

                        buffer.add(
                            observations,
                            res.actions, res.action_logits, res.log_prob_actions,
                            res.values,
                            masks, rnn_states,
                            rewards, dones,
                        )

                        with timing.timeit('obs'):
                            # TODO: can we do it faster? Maybe in Pytorch
                            observations = self._preprocess_observations(new_obs)
                        rnn_states = res.rnn_states

                        num_steps += num_env_steps(infos)

                    # last step values are required for TD-return calculation
                    next_values = self.actor_critic(observations, rnn_states, self._get_masks(dones)).values
                    buffer.values.append(next_values)

                    self.env_steps += num_steps

                with timing.timeit('finalize'):
                    # calculate discounted returns and GAE
                    buffer.finalize_batch(self.cfg.gamma, self.cfg.gae_lambda)

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
