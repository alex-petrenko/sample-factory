import torch
from torch import nn

from sample_factory.model.encoder import Encoder, make_img_encoder
from sample_factory.model.model_utils import model_device
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log
from sf_examples.dmlab.dmlab30 import DMLAB_INSTRUCTIONS, DMLAB_VOCABULARY_SIZE


class DmlabEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.basic_encoder = make_img_encoder(cfg, obs_space["obs"])
        self.encoder_out_size = self.basic_encoder.get_out_size()

        # same as IMPALA paper
        self.embedding_size = 20
        self.instructions_lstm_units = 64
        self.instructions_lstm_layers = 1

        padding_idx = 0
        self.word_embedding = nn.Embedding(
            num_embeddings=DMLAB_VOCABULARY_SIZE, embedding_dim=self.embedding_size, padding_idx=padding_idx
        )

        self.instructions_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.instructions_lstm_units,
            num_layers=self.instructions_lstm_layers,
            batch_first=True,
        )

        # learnable initial state?
        # initial_hidden_values = torch.normal(0, 1, size=(self.instructions_lstm_units, ))
        # self.lstm_h0 = nn.Parameter(initial_hidden_values, requires_grad=True)
        # self.lstm_c0 = nn.Parameter(initial_hidden_values, requires_grad=True)

        self.encoder_out_size += self.instructions_lstm_units
        log.debug("DMLab policy head output size: %r", self.encoder_out_size)

        self.cpu_device = torch.device("cpu")

    def model_to_device(self, device):
        self.to(device)
        self.word_embedding.to(self.cpu_device)
        self.instructions_lstm.to(self.cpu_device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        if input_tensor_name == DMLAB_INSTRUCTIONS:
            return self.cpu_device
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        if input_tensor_name == DMLAB_INSTRUCTIONS:
            return torch.int64
        return torch.float32

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict["obs"])

        with torch.no_grad():
            instr = obs_dict[DMLAB_INSTRUCTIONS]
            instr_lengths = (instr != 0).sum(axis=1)
            instr_lengths = torch.clamp(instr_lengths, min=1)
            max_instr_len = torch.max(instr_lengths).item()
            instr = instr[:, :max_instr_len]

        instr_embed = self.word_embedding(instr)
        instr_packed = torch.nn.utils.rnn.pack_padded_sequence(
            instr_embed,
            instr_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        rnn_output, _ = self.instructions_lstm(instr_packed)
        rnn_outputs, sequence_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        first_dim_idx = torch.arange(rnn_outputs.shape[0])
        last_output_idx = sequence_lengths - 1
        last_outputs = rnn_outputs[first_dim_idx, last_output_idx]

        last_outputs = last_outputs.to(x.device)  # for some reason this is very slow

        x = torch.cat((x, last_outputs), dim=1)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_dmlab_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    return DmlabEncoder(cfg, obs_space)
