import torch
from torch import nn

from sample_factory.algorithms.appo.model_utils import create_encoder, create_core, \
    ActionParameterizationContinuousNonAdaptiveStddev, \
    ActionParameterizationDefault, normalize_obs
from sample_factory.algorithms.utils.action_distributions import sample_actions_log_probs, is_continuous_action_space
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import AttrDict


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
            else:
                # LSTMs and GRUs initialize themselves
                # should we use orthogonal/xavier for LSTM cells as well?
                # I never noticed much difference between different initialization schemes, and here it seems safer to
                # go with default initialization,
                pass
        elif self.cfg.policy_initialization == 'xavier_uniform':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
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
