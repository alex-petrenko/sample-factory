from torch import nn

from algorithms.appo.model_utils import create_encoder, create_core
from algorithms.utils.action_distributions import calc_num_logits, sample_actions_log_probs, get_action_distribution
from utils.timing import Timing
from utils.utils import AttrDict


class _ActorCritic(nn.Module):
    def __init__(self, encoder, core, action_space, cfg, timing):
        super().__init__()

        self.cfg = cfg
        self.action_space = action_space

        self.timing = timing

        self.encoder = encoder

        self.core = core

        self.critic_linear = nn.Linear(self.cfg.hidden_size, 1)
        self.distribution_linear = nn.Linear(
            self.cfg.hidden_size, calc_num_logits(self.action_space),
        )

        self.apply(self.initialize_weights)

        self.train()

    def forward_head(self, obs_dict):
        x = self.encoder(obs_dict)
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

    def model_to_device(self, device):
        self.to(device)
        self.encoder.model_to_device(device)

    def device_and_type_for_input_tensor(self, input_tensor_name):
        return self.encoder.device_and_type_for_input_tensor(input_tensor_name)

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


def create_actor_critic(cfg, obs_space, action_space, timing=None):
    if timing is None:
        timing = Timing()

    encoder = create_encoder(cfg, obs_space, timing)
    core = create_core(cfg, encoder.encoder_out_size)

    return _ActorCritic(encoder, core, action_space, cfg, timing)
