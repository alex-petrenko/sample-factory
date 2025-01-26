from sf_examples.nethack.models.chaotic_dwarf import ChaoticDwarvenGPT5
from sf_examples.nethack.models.scaled import ScaledNet
from sf_examples.nethack.models.simba import SimBaActorEncoder, SimBaCriticEncoder
from sf_examples.nethack.models.vit import ViTActorEncoder, ViTCriticEncoder

MODELS = [SimBaActorEncoder, SimBaCriticEncoder, ViTActorEncoder, ViTCriticEncoder, ChaoticDwarvenGPT5, ScaledNet]
MODELS_LOOKUP = {c.__name__: c for c in MODELS}
