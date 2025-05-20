from typing import Callable

from sample_factory.model.actor_critic import ActorCritic, default_make_actor_critic_func
from sample_factory.model.core import ModelCore, default_make_core_func
from sample_factory.model.decoder import Decoder, default_make_decoder_func
from sample_factory.model.encoder import Encoder, default_make_encoder_func
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.utils.utils import log

MakeActorCriticFunc = Callable[[Config, ObsSpace, ActionSpace], ActorCritic]
MakeEncoderFunc = Callable[[Config, ObsSpace], Encoder]
MakeCoreFunc = Callable[[Config, int], ModelCore]
MakeDecoderFunc = Callable[[Config, int], Decoder]


class ModelFactory:
    def __init__(self):
        """
        Optional custom functions for creating parts of the model (encoders, decoders, etc.), or
        even overriding the entire actor-critic with a custom model.
        """

        self.make_actor_critic_func: MakeActorCriticFunc = default_make_actor_critic_func

        # callables user can specify to generate parts of the policy
        # the computational graph structure is:
        # observations -> encoder -> core -> decoder -> actions
        self.make_model_encoder_func: MakeEncoderFunc = default_make_encoder_func
        self.make_model_core_func: MakeCoreFunc = default_make_core_func
        self.make_model_decoder_func: MakeDecoderFunc = default_make_decoder_func

    def register_actor_critic_factory(self, make_actor_critic_func: MakeActorCriticFunc):
        """
        Override the default actor-critic with a custom model.
        """
        log.debug(f"register_actor_critic_factory: {make_actor_critic_func}")
        self.make_actor_critic_func = make_actor_critic_func

    def register_encoder_factory(self, make_model_encoder_func: MakeEncoderFunc):
        """
        Override the default encoder with a custom model.
        The computational graph structure is: observations -> encoder -> core -> decoder -> actions
        """
        log.debug(f"register_encoder_factory: {make_model_encoder_func}")
        self.make_model_encoder_func = make_model_encoder_func

    def register_model_core_factory(self, make_model_core_func: MakeCoreFunc):
        """
        Override the default core with a custom model.
        The computational graph structure is: observations -> encoder -> core -> decoder -> actions
        """
        log.debug(f"register_model_core_factory: {make_model_core_func}")
        self.make_model_core_func = make_model_core_func

    def register_decoder_factory(self, make_model_decoder_func: MakeDecoderFunc):
        """
        Override the default decoder with a custom model.
        The computational graph structure is: observations -> encoder -> core -> decoder -> actions
        """
        log.debug(f"register_decoder_factory: {make_model_decoder_func}")
        self.make_model_decoder_func = make_model_decoder_func
