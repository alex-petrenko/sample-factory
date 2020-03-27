from torch import nn

from algorithms.appo.model_utils import nonlinearity, create_encoder, create_core
from algorithms.utils.action_distributions import calc_num_logits, sample_actions_log_probs, get_action_distribution
from utils.utils import AttrDict


def fc_after_encoder_size(cfg):
    return cfg.hidden_size  # make configurable?


class _ActorCritic(nn.Module):
    def __init__(self, encoder, core, action_space, cfg):
        super().__init__()

        self.cfg = cfg
        self.action_space = action_space

        self.encoder = encoder

        self.linear_after_enc = None
        if cfg.fc_after_encoder:
            self.linear_after_enc = nn.Sequential(
                nn.Linear(self.encoder.encoder_out_size, fc_after_encoder_size(cfg)),
                nonlinearity(cfg),
            )

        self.core = core

        self.critic_linear = nn.Linear(self.cfg.hidden_size, 1)
        self.distribution_linear = nn.Linear(
            self.cfg.hidden_size, calc_num_logits(self.action_space),
        )

        self.apply(self.initialize_weights)

        self.train()

    def forward_head(self, obs_dict):
        x = self.encoder(obs_dict)
        if self.linear_after_enc is not None:
            x = self.linear_after_enc(x)
        return x

    def forward_core(self, head_output, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(self, core_output, with_action_distribution=False):
        values = self.critic_linear(core_output)
        action_logits = self.distribution_linear(core_output)

        dist = get_action_distribution(self.action_space, raw_logits=action_logits)

        # for non-trivial action spaces it is faster to do these together
        actions, log_prob_actions = sample_actions_log_probs(dist)

        result = AttrDict(dict(
            actions=actions,
            action_logits=action_logits,
            log_prob_actions=log_prob_actions,
            values=values,
        ))

        if with_action_distribution:
            result.action_distribution = dist

        return result

    def forward(self, obs_dict, rnn_states):
        x = self.forward_head(obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x)
        result.rnn_states = new_rnn_states
        return result

    @staticmethod
    def initialize_weights(layer):
        """TODO: test xavier initialization"""

        if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
            nn.init.orthogonal_(layer.weight.data, gain=1)
            layer.bias.data.fill_(0)
        elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:  # TODO: test for LSTM
            nn.init.orthogonal_(layer.weight_ih, gain=1)
            nn.init.orthogonal_(layer.weight_hh, gain=1)
            layer.bias_ih.data.fill_(0)
            layer.bias_hh.data.fill_(0)
        else:
            pass


def create_actor_critic(cfg, obs_space, action_space):
    encoder = create_encoder(cfg, obs_space)

    if cfg.fc_after_encoder:
        core_input_size = fc_after_encoder_size(cfg)
    else:
        core_input_size = encoder.encoder_out_size

    core = create_core(cfg, core_input_size)

    return _ActorCritic(encoder, core, action_space, cfg)
