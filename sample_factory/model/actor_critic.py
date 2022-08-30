from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from sample_factory.algo.utils.action_distributions import is_continuous_action_space, sample_actions_log_probs
from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace, running_mean_std_summaries
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.configurable import Configurable
from sample_factory.model.action_parameterization import (
    ActionParameterizationContinuousNonAdaptiveStddev,
    ActionParameterizationDefault,
)
from sample_factory.model.model_utils import model_device
from sample_factory.utils.normalize import ObservationNormalizer
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace


class ActorCritic(nn.Module, Configurable):
    def __init__(self, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
        nn.Module.__init__(self)
        Configurable.__init__(self, cfg)
        self.action_space = action_space
        self.encoders = []

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

    def get_action_parameterization(self, decoder_output_size: int):
        if not self.cfg.adaptive_stddev and is_continuous_action_space(self.action_space):
            action_parameterization = ActionParameterizationContinuousNonAdaptiveStddev(
                self.cfg,
                decoder_output_size,
                self.action_space,
            )
        else:
            action_parameterization = ActionParameterizationDefault(self.cfg, decoder_output_size, self.action_space)

        return action_parameterization

    def model_to_device(self, device):
        for module in self.children():
            # allow parts of encoders/decoders to be on different devices
            # (i.e. text-encoding LSTM for DMLab is faster on CPU)
            if hasattr(module, "model_to_device"):
                module.model_to_device(device)
            else:
                module.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        device = self.encoders[0].device_for_input_tensor(input_tensor_name)
        if device is None:
            device = model_device(self)
        return device

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

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward_core(self, head_output, rnn_states):
        raise NotImplementedError()

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        raise NotImplementedError()

    def forward(self, normalized_obs_dict, rnn_states, values_only: bool = False) -> TensorDict:
        raise NotImplementedError()


class ActorCriticSharedWeights(ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        super().__init__(obs_space, action_space, cfg)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.encoders = [self.encoder]  # a single shared encoder

        self.core = model_factory.make_model_core_func(cfg, self.encoder.get_out_size())

        self.decoder = model_factory.make_model_decoder_func(cfg, self.core.get_out_size())
        decoder_out_size: int = self.decoder.get_out_size()

        self.critic_linear = nn.Linear(decoder_out_size, 1)
        self.action_parameterization = self.get_action_parameterization(decoder_out_size)

        self.apply(self.initialize_weights)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = self.action_parameterization(decoder_output)

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result


class ActorCriticSeparateWeights(ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        super().__init__(obs_space, action_space, cfg)

        self.actor_encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.actor_core = model_factory.make_model_core_func(cfg, self.actor_encoder.get_out_size())

        self.critic_encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.critic_core = model_factory.make_model_core_func(cfg, self.critic_encoder.get_out_size())

        self.encoders = [self.actor_encoder, self.critic_encoder]
        self.cores = [self.actor_core, self.critic_core]

        self.core_func = self._core_rnn if self.cfg.use_rnn else self._core_empty

        self.actor_decoder = model_factory.make_model_decoder_func(cfg, self.actor_core.get_out_size())
        self.critic_decoder = model_factory.make_model_decoder_func(cfg, self.critic_core.get_out_size())
        self.decoders = [self.actor_decoder, self.critic_decoder]

        self.critic_linear = nn.Linear(self.critic_decoder.get_out_size(), 1)
        self.action_parameterization = self.get_action_parameterization(self.critic_decoder.get_out_size())

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

    def forward_head(self, normalized_obs_dict: Dict):
        head_outputs = []
        for enc in self.encoders:
            head_outputs.append(enc(normalized_obs_dict))

        return torch.cat(head_outputs, dim=1)

    def forward_core(self, head_output, rnn_states):
        return self.core_func(head_output, rnn_states)

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        core_outputs = core_output.chunk(len(self.cores), dim=1)

        # second core output corresponds to the critic
        critic_decoder_output = self.critic_decoder(core_outputs[1])
        values = self.critic_linear(critic_decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            # this can be further optimized - we don't need to calculate actor head/core just to get values
            return result

        # first core output corresponds to the actor
        actor_decoder_output = self.actor_decoder(core_outputs[0])
        action_distribution_params, self.last_action_distribution = self.action_parameterization(actor_decoder_output)

        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result


def default_make_actor_critic_func(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    from sample_factory.algo.utils.context import global_model_factory

    model_factory = global_model_factory()

    if cfg.actor_critic_share_weights:
        return ActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
    else:
        return ActorCriticSeparateWeights(model_factory, obs_space, action_space, cfg)


def create_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    # check if user specified custom actor/critic creation function
    from sample_factory.algo.utils.context import global_model_factory

    make_actor_critic_func = global_model_factory().make_actor_critic_func
    return make_actor_critic_func(cfg, obs_space, action_space)
