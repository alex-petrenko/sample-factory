from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log


def get_rnn_size(cfg):
    if cfg.use_rnn:
        size = cfg.rnn_size * cfg.rnn_num_layers
    else:
        size = 1

    if cfg.rnn_type == "lstm":
        size *= 2

    if not cfg.actor_critic_share_weights:
        # actor and critic need separate states
        size *= 2

    return size


def nonlinearity(cfg: Config, inplace: bool = False) -> nn.Module:
    if cfg.nonlinearity == "elu":
        return nn.ELU(inplace=inplace)
    elif cfg.nonlinearity == "relu":
        return nn.ReLU(inplace=inplace)
    elif cfg.nonlinearity == "tanh":
        return nn.Tanh()
    else:
        raise Exception(f"Unknown {cfg.nonlinearity=}")


def fc_layer(in_features: int, out_features: int, bias=True, spec_norm=False) -> nn.Module:
    layer = nn.Linear(in_features, out_features, bias)
    if spec_norm:
        layer = spectral_norm(layer)

    return layer


def create_mlp(layer_sizes: List[int], input_size: int, activation: nn.Module) -> nn.Module:
    """Sequential fully connected layers."""
    layers = []
    for i, size in enumerate(layer_sizes):
        layers.extend([fc_layer(input_size, size), activation])
        input_size = size

    if len(layers) > 0:
        return nn.Sequential(*layers)
    else:
        return nn.Identity()


class ModelModule(nn.Module, Configurable):
    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        Configurable.__init__(self, cfg)

    def get_out_size(self):
        raise NotImplementedError()


def model_device(model: nn.Module) -> Optional[torch.device]:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _legacy_key_candidates(key: str) -> List[str]:
    """Current names a parameter saved under `key` by an older Sample Factory version may have."""
    # 521b7edc ("V75") renamed ConvEncoderImpl.fc_layers to mlp_layers
    renamed_fc = key.replace(".enc.fc_layers.", ".enc.mlp_layers.")
    candidates = [renamed_fc]

    # MultiInputEncoder (#198) moved the single encoder under encoder.encoders.obs
    # (checkpoints older than both refactors need both renames)
    prefix = "encoder."
    for k in (key, renamed_fc):
        if k.startswith(prefix):
            candidates.append("encoder.encoders.obs." + k[len(prefix) :])

    return candidates


def upgrade_legacy_state_dict(state_dict: Dict[str, Tensor], model: nn.Module) -> Dict[str, Tensor]:
    """
    Rename parameters in a checkpoint saved before the 2022 model refactors (e.g. the pretrained models on the HF Hub),
    so it can be loaded into `model`. A key is only renamed if the model expects the new name and not the old one,
    so up-to-date checkpoints are returned unchanged and genuine mismatches are still reported by load_state_dict().
    """
    expected_keys = model.state_dict().keys()
    upgraded = OrderedDict()
    renamed = 0
    for key, value in state_dict.items():
        if key not in expected_keys:
            for candidate in _legacy_key_candidates(key):
                if candidate in expected_keys and candidate not in state_dict:
                    key = candidate
                    renamed += 1
                    break
        upgraded[key] = value

    if not renamed:
        return state_dict

    log.warning(f"Checkpoint was saved by an older version of Sample Factory, renamed {renamed} parameters")
    # torch.nn.Module.load_state_dict() reads per-module versions from this attribute, keep it
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        upgraded._metadata = metadata
    return upgraded
