from sf_examples.nethack.models.vit import ViTActorEncoder, ViTCriticEncoder

MODELS = [ViTActorEncoder, ViTCriticEncoder]
MODELS_LOOKUP = {c.__name__: c for c in MODELS}
