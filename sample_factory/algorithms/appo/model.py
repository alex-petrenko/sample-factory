from typing import Tuple

import gym
import torch
from torch import nn
from torch.nn import functional as F

from sample_factory.algorithms.appo.model_utils import create_encoder, create_core, ActionParameterizationContinuousNonAdaptiveStddev, \
    ActionParameterizationDefault, normalize_obs
from sample_factory.algorithms.utils.action_distributions import sample_actions_log_probs, is_continuous_action_space, calc_num_actions, calc_num_logits
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import AttrDict


class CPCA(nn.Module):
    def __init__(self, cfg, action_space):
        super().__init__()
        self.k: int = cfg.cpc_forward_steps
        self.time_subsample: int = cfg.cpc_time_subsample
        self.forward_subsample: int = cfg.cpc_forward_subsample
        self.hidden_size: int = cfg.hidden_size
        self.num_actions: int = calc_num_actions(action_space)
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_sizes = [action_space.n]
        else:
            self.action_sizes = [space.n for space in action_space.spaces]

        self.rnn = nn.GRU(32 * self.num_actions, cfg.hidden_size)
        self.action_embed = nn.Embedding(calc_num_logits(action_space), 32)

        self.predictor = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, 1),
        )

    def embed_actions(self, actions):
        embedded = []
        offset = 0
        for i in range(self.num_actions):
            embedded.append(self.action_embed(actions[..., i] + offset).squeeze())
            offset += self.action_sizes[i]

        return torch.cat(embedded, -1)

    def _build_unfolded(self, x, k: int):
        return (
            torch.cat((x, x.new_zeros(x.size(0), k, x.size(2))), 1)
            .unfold(1, size=k, step=1)
            .permute(3, 0, 1, 2)
        )

    def _build_mask_and_subsample(
        self, not_dones
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        t = not_dones.size(1)

        not_dones_unfolded = self._build_unfolded(
            not_dones[:, :-1].to(dtype=torch.bool), self.k
        )

        time_subsample = torch.randperm(
            t - 1, device=not_dones.device, dtype=torch.long
        )[0:self.time_subsample]

        forward_mask = (
            torch.cumprod(not_dones_unfolded.index_select(2, time_subsample), dim=0)
            .to(dtype=torch.bool)
            .flatten(1, 2)
        )

        max_k = forward_mask.flatten(1).any(-1).nonzero().max().item() + 1

        unroll_subsample = torch.randperm(max_k, dtype=torch.long)[0:self.forward_subsample]

        max_k = unroll_subsample.max().item() + 1

        unroll_subsample = unroll_subsample.to(device=not_dones.device)
        forward_mask = forward_mask.index_select(0, unroll_subsample)

        return forward_mask, unroll_subsample, time_subsample, max_k

    def forward(self, actions, not_dones, rnn_inputs, rnn_outputs):
        n = actions.size(0)
        t = actions.size(1)

        mask_res = self._build_mask_and_subsample(not_dones)
        (forward_mask, unroll_subsample, time_subsample, max_k,) = mask_res

        actions = self.embed_actions(actions.long())
        actions_unfolded = self._build_unfolded(actions[:, :-1], max_k).index_select(
            2, time_subsample
        )

        rnn_outputs_subsampled = rnn_outputs[:, :-1].index_select(1, time_subsample)
        forward_preds, _ = self.rnn(
            actions_unfolded.contiguous().flatten(1, 2),
            rnn_outputs_subsampled.contiguous().view(1, -1, self.hidden_size),
        )
        forward_preds = forward_preds.index_select(0, unroll_subsample)
        forward_targets = self._build_unfolded(rnn_inputs[:, 1:], max_k)
        forward_targets = (
            forward_targets.index_select(2, time_subsample)
            .index_select(0, unroll_subsample)
            .flatten(1, 2)
        )

        positives = self.predictor(torch.cat((forward_preds, forward_targets), dim=-1))
        positive_loss = F.binary_cross_entropy_with_logits(
            positives, torch.broadcast_tensors(positives, positives.new_ones(()))[1], reduction="none"
        )
        positive_loss = torch.masked_select(positive_loss, forward_mask).mean()

        forward_negatives = torch.randint(
            0, n * t, size=(self.forward_subsample * self.time_subsample * n * 20,), dtype=torch.long, device=actions.device
        )
        forward_negatives = (
            rnn_inputs.flatten(0, 1)
            .index_select(0, forward_negatives)
            .view(self.forward_subsample, self.time_subsample * n, 20, -1)
        )
        negatives = self.predictor(
            torch.cat(
                (
                    forward_preds.view(self.forward_subsample, self.time_subsample * n, 1, -1)
                        .expand(-1, -1, 20, -1),
                    forward_negatives,
                ),
                dim=-1,
            )
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives, torch.broadcast_tensors(negatives, negatives.new_zeros(()))[1], reduction="none"
        )
        negative_loss = torch.masked_select(
            negative_loss, forward_mask.unsqueeze(2)
        ).mean()

        return 0.1 * (positive_loss + negative_loss)


class _ActorCriticBase(nn.Module):
    def __init__(self, action_space, cfg, timing):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space
        self.timing = timing
        self.encoders = []
        self.cores = []

    def get_action_parameterization(self, core_output_size):
        if not self.cfg.adaptive_stddev and is_continuous_action_space(self.action_space):
            action_parameterization = ActionParameterizationContinuousNonAdaptiveStddev(
                self.cfg, core_output_size, self.action_space,
            )
        else:
            action_parameterization = ActionParameterizationDefault(self.cfg, core_output_size, self.action_space)

        return action_parameterization

    def model_to_device(self, device):
        self.to(device)
        for e in self.encoders:
            e.model_to_device(device)

    def device_and_type_for_input_tensor(self, input_tensor_name):
        return self.encoders[0].device_and_type_for_input_tensor(input_tensor_name)

    def initialize_weights(self, layer):
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        gain = self.cfg.policy_init_gain

        if self.cfg.policy_initialization == 'orthogonal':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:  # TODO: test for LSTM
                nn.init.orthogonal_(layer.weight_ih, gain=gain)
                nn.init.orthogonal_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
            else:
                pass
        elif self.cfg.policy_initialization == 'xavier_uniform':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:
                nn.init.xavier_uniform_(layer.weight_ih, gain=gain)
                nn.init.xavier_uniform_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
            else:
                pass


class _ActorCriticSharedWeights(_ActorCriticBase):
    def __init__(self, make_encoder, make_core, action_space, cfg, timing):
        super().__init__(action_space, cfg, timing)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = make_encoder()
        self.encoders = [self.encoder]

        self.core = make_core(self.encoder)
        self.cores = [self.core]

        core_out_size = self.core.get_core_out_size()
        self.critic_linear = nn.Linear(core_out_size, 1)

        self.action_parameterization = self.get_action_parameterization(core_out_size)

        self.apply(self.initialize_weights)
        self.train()  # eval() for inference?

    def forward_head(self, obs_dict):
        normalize_obs(obs_dict, self.cfg)
        x = self.encoder(obs_dict)
        return x

    def forward_core(self, head_output, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(self, core_output, with_action_distribution=False):
        values = self.critic_linear(core_output)

        action_distribution_params, action_distribution = self.action_parameterization(core_output)

        # for non-trivial action spaces it is faster to do these together
        actions, log_prob_actions = sample_actions_log_probs(action_distribution)

        result = AttrDict(dict(
            actions=actions,
            action_logits=action_distribution_params,  # perhaps `action_logits` is not the best name here since we now support continuous actions
            log_prob_actions=log_prob_actions,
            values=values,
        ))

        if with_action_distribution:
            result.action_distribution = action_distribution

        return result

    def forward(self, obs_dict, rnn_states, with_action_distribution=False):
        x = self.forward_head(obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, with_action_distribution=with_action_distribution)
        result.rnn_states = new_rnn_states
        return result


class _ActorCriticSeparateWeights(_ActorCriticBase):
    def __init__(self, make_encoder, make_core, action_space, cfg, timing):
        super().__init__(action_space, cfg, timing)

        self.actor_encoder = make_encoder()
        self.actor_core = make_core(self.actor_encoder)

        self.critic_encoder = make_encoder()
        self.critic_core = make_core(self.critic_encoder)

        self.encoders = [self.actor_encoder, self.critic_encoder]
        self.cores = [self.actor_core, self.critic_core]

        self.core_func = self._core_rnn if self.cfg.use_rnn else self._core_empty

        self.critic_linear = nn.Linear(self.critic_core.get_core_out_size(), 1)

        self.action_parameterization = self.get_action_parameterization(self.critic_core.get_core_out_size())

        self.apply(self.initialize_weights)

        self.train()

    def _core_rnn(self, head_output, rnn_states):
        """
        This is actually pretty slow due to all these split and cat operations.
        Consider using shared weights when training RNN policies.
        """

        num_cores = len(self.cores)
        head_outputs_split = head_output.chunk(num_cores, dim=1)
        rnn_states_split = rnn_states.chunk(num_cores, dim=1)

        outputs, new_rnn_states = [], []
        for i, c in enumerate(self.cores):
            output, new_rnn_state = c(head_outputs_split[i], rnn_states_split[i])
            outputs.append(output)
            new_rnn_states.append(new_rnn_state)

        outputs = torch.cat(outputs, dim=1)
        new_rnn_states = torch.cat(new_rnn_states, dim=1)
        return outputs, new_rnn_states

    @staticmethod
    def _core_empty(head_output, fake_rnn_states):
        """Optimization for the feed-forward case."""
        return head_output, fake_rnn_states

    def forward_head(self, obs_dict):
        normalize_obs(obs_dict, self.cfg)

        head_outputs = []
        for e in self.encoders:
            head_outputs.append(e(obs_dict))

        return torch.cat(head_outputs, dim=1)

    def forward_core(self, head_output, rnn_states):
        return self.core_func(head_output, rnn_states)

    def forward_tail(self, core_output, with_action_distribution=False):
        core_outputs = core_output.chunk(len(self.cores), dim=1)

        # first core output corresponds to the actor
        action_distribution_params, action_distribution = self.action_parameterization(core_outputs[0])
        # for non-trivial action spaces it is faster to do these together
        actions, log_prob_actions = sample_actions_log_probs(action_distribution)

        # second core output corresponds to the critic
        values = self.critic_linear(core_outputs[1])

        result = AttrDict(dict(
            actions=actions,
            action_logits=action_distribution_params,
            log_prob_actions=log_prob_actions,
            values=values,
        ))

        if with_action_distribution:
            result.action_distribution = action_distribution

        return result

    def forward(self, obs_dict, rnn_states, with_action_distribution=False):
        x = self.forward_head(obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, with_action_distribution=with_action_distribution)
        result.rnn_states = new_rnn_states
        return result


def create_actor_critic(cfg, obs_space, action_space, timing=None):
    if timing is None:
        timing = Timing()

    def make_encoder():
        return create_encoder(cfg, obs_space, timing)

    def make_core(encoder):
        return create_core(cfg, encoder.get_encoder_out_size())

    if cfg.actor_critic_share_weights:
        return _ActorCriticSharedWeights(make_encoder, make_core, action_space, cfg, timing)
    else:
        return _ActorCriticSeparateWeights(make_encoder, make_core, action_space, cfg, timing)
