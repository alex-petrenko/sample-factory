from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from sample_factory.algo.utils.action_distributions import is_continuous_action_space, sample_actions_log_probs
from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace, running_mean_std_summaries
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import (
    ActionParameterizationContinuousNonAdaptiveStddev,
    ActionParameterizationDefault,
    create_core,
    create_encoder,
)
from sample_factory.utils.normalize import ObservationNormalizer
from sample_factory.utils.timing import Timing


class _ActorCriticBase(nn.Module):
    def __init__(self, obs_space, action_space, cfg, timing):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space
        self.timing = timing
        self.encoders = []
        self.cores = []

        # we make normalizers a part of the model, so we can use the same infrastructure
        # to load/save the state of the normalizer (running mean and stddev statistics)
        self.obs_normalizer: ObservationNormalizer = ObservationNormalizer(obs_space, cfg)

        self.returns_normalizer: Optional[RunningMeanStdInPlace] = None
        if cfg.normalize_returns:
            returns_shape = (1,)  # it's actually a single scalar but we use 1D shape for the normalizer
            self.returns_normalizer = RunningMeanStdInPlace(returns_shape)
            # comment this out for debugging (i.e. to be able to step through normalizer code)
            self.returns_normalizer = torch.jit.script(self.returns_normalizer)

        self.last_action_distribution = None  # to be populated after each forward step

    def get_action_parameterization(self, core_output_size):
        if not self.cfg.adaptive_stddev and is_continuous_action_space(self.action_space):
            action_parameterization = ActionParameterizationContinuousNonAdaptiveStddev(
                self.cfg,
                core_output_size,
                self.action_space,
            )
        else:
            action_parameterization = ActionParameterizationDefault(self.cfg, core_output_size, self.action_space)

        return action_parameterization

    def model_to_device(self, device):
        self.to(device)
        for e in self.encoders:
            e.model_to_device(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return self.encoders[0].device_for_input_tensor(input_tensor_name)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return self.encoders[0].type_for_input_tensor(input_tensor_name)

    def initialize_weights(self, layer):
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        gain = self.cfg.policy_init_gain

        if hasattr(layer, "bias") and isinstance(layer.bias, torch.nn.parameter.Parameter):
            layer.bias.data.fill_(0)

        if self.cfg.policy_initialization == "orthogonal":
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
            else:
                # LSTMs and GRUs initialize themselves
                # should we use orthogonal/xavier for LSTM cells as well?
                # I never noticed much difference between different initialization schemes, and here it seems safer to
                # go with default initialization,
                pass
        elif self.cfg.policy_initialization == "xavier_uniform":
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
            else:
                pass
        elif self.cfg.policy_initialization == "torch_default":
            # do nothing
            pass

    def normalize_obs(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.obs_normalizer(obs)

    def summaries(self) -> Dict:
        # Can add more summaries here, like weights statistics
        s = self.obs_normalizer.summaries()
        if self.returns_normalizer is not None:
            for k, v in running_mean_std_summaries(self.returns_normalizer).items():
                s[f"returns_{k}"] = v
        return s

    def action_distribution(self):
        return self.last_action_distribution

    def _maybe_sample_actions(self, sample_actions: bool, result: TensorDict) -> None:
        if sample_actions:
            # for non-trivial action spaces it is faster to do these together
            actions, result["log_prob_actions"] = sample_actions_log_probs(self.last_action_distribution)
            assert actions.dim() == 2  # TODO: remove this once we test everything
            result["actions"] = actions.squeeze(dim=1)


class _ActorCriticSharedWeights(_ActorCriticBase):
    def __init__(self, make_encoder, make_core, obs_space, action_space, cfg, timing):
        super().__init__(obs_space, action_space, cfg, timing)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = make_encoder()
        self.encoders = [self.encoder]

        self.core = make_core(self.encoder)
        self.cores = [self.core]

        core_out_size = self.core.get_core_out_size()
        self.critic_linear = nn.Linear(core_out_size, 1)

        self.action_parameterization = self.get_action_parameterization(core_out_size)

        self.apply(self.initialize_weights)

    def forward_head(self, normalized_obs_dict):
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def calc_value(self, core_output):
        values = self.critic_linear(core_output).squeeze()
        return values

    def forward_tail(self, core_output, sample_actions=True) -> TensorDict:
        values = self.calc_value(core_output)

        action_distribution_params, self.last_action_distribution = self.action_parameterization(core_output)

        result = TensorDict(
            action_logits=action_distribution_params,
            # TODO! `action_logits` is not the best name here since we now support continuous actions
            values=values,
        )

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)

        if values_only:
            return TensorDict(values=self.calc_value(x))
        else:
            result = self.forward_tail(x)
            result["new_rnn_states"] = new_rnn_states
            return result


class _ActorCriticSeparateWeights(_ActorCriticBase):
    def __init__(self, make_encoder, make_core, obs_space, action_space, cfg, timing):
        super().__init__(obs_space, action_space, cfg, timing)

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

    def forward_head(self, normalized_obs_dict):
        head_outputs = []
        for e in self.encoders:
            head_outputs.append(e(normalized_obs_dict))

        return torch.cat(head_outputs, dim=1)

    def forward_core(self, head_output, rnn_states):
        return self.core_func(head_output, rnn_states)

    def forward_tail(self, core_output, sample_actions=True) -> TensorDict:
        core_outputs = core_output.chunk(len(self.cores), dim=1)

        # first core output corresponds to the actor
        action_distribution_params, self.last_action_distribution = self.action_parameterization(core_outputs[0])

        # second core output corresponds to the critic
        values = self.critic_linear(core_outputs[1]).squeeze()

        result = TensorDict(
            action_logits=action_distribution_params,
            values=values,
        )

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)

        if values_only:
            # TODO: this can be further optimized (we don't need to calculate the entire policy head to generate values)
            # also currently untested
            core_outputs = x.chunk(len(self.cores), dim=1)
            # second core output corresponds to the critic
            values = self.critic_linear(core_outputs[1]).squeeze()
            return TensorDict(values=values)
        else:
            result = self.forward_tail(x)
            result["new_rnn_states"] = new_rnn_states
            return result


def create_actor_critic(cfg, obs_space, action_space, timing=None):
    if timing is None:
        timing = Timing()

    def make_encoder():
        return create_encoder(cfg, obs_space, timing)

    def make_core(encoder):
        return create_core(cfg, encoder.get_encoder_out_size())

    if cfg.actor_critic_share_weights:
        return _ActorCriticSharedWeights(make_encoder, make_core, obs_space, action_space, cfg, timing)
    else:
        return _ActorCriticSeparateWeights(make_encoder, make_core, obs_space, action_space, cfg, timing)
