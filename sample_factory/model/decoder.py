from abc import ABC
from typing import List

import torch

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.model_utils import ModelModule, create_mlp, nonlinearity
from sample_factory.utils.typing import Config


class Decoder(ModelModule, ABC):
    pass


class MlpDecoder(Decoder):
    def __init__(self, cfg: Config, decoder_input_size: int):
        super().__init__(cfg)
        self.core_input_size = decoder_input_size
        decoder_layers: List[int] = cfg.decoder_mlp_layers
        activation = nonlinearity(cfg)
        self.mlp = create_mlp(decoder_layers, decoder_input_size, activation)
        if len(decoder_layers) > 0:
            self.mlp = torch.jit.script(self.mlp)

        self.decoder_out_size = calc_num_elements(self.mlp, (decoder_input_size,))

    def forward(self, core_output):
        return self.mlp(core_output)

    def get_out_size(self):
        return self.decoder_out_size


def default_make_decoder_func(cfg: Config, core_input_size: int) -> Decoder:
    return MlpDecoder(cfg, core_input_size)
