from sf_examples.nethack.models.chaotic_dwarf import ChaoticDwarvenGPT5
from sf_examples.nethack.models.scaled import ScaledNet
from sf_examples.nethack.models.simba import SimbaActorEncoder, SimbaCriticEncoder

MODELS = [
    ChaoticDwarvenGPT5,
    ScaledNet,
    SimbaActorEncoder,
    SimbaCriticEncoder,
]
MODELS_LOOKUP = {c.__name__: c for c in MODELS}
