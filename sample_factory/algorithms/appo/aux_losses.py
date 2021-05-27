from typing import Tuple

import gym
import torch
from torch import nn
from torch.nn import functional as F

from sample_factory.algorithms.utils.action_distributions import calc_num_actions, calc_num_logits


class CPCA(nn.Module):
    def __init__(self, cfg, action_space):
        super().__init__()
        self.k: int = cfg.cpc_forward_steps
        self.time_subsample: int = cfg.cpc_time_subsample
        self.forward_subsample: int = cfg.cpc_forward_subsample
        self.hidden_size: int = cfg.hidden_size
        self.num_actions: int = calc_num_actions(action_space)
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_sizes = [action_space.n]
        else:
            self.action_sizes = [space.n for space in action_space.spaces]

        self.rnn = nn.GRU(32 * self.num_actions, cfg.hidden_size)
        self.action_embed = nn.Embedding(calc_num_logits(action_space), 32)

        self.predictor = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, 1),
        )

    def embed_actions(self, actions):
        embedded = []
        offset = 0
        for i in range(self.num_actions):
            embedded.append(self.action_embed(actions[..., i] + offset).squeeze())
            offset += self.action_sizes[i]

        return torch.cat(embedded, -1)

    def _build_unfolded(self, x, k: int):
        return (
            torch.cat((x, x.new_zeros(x.size(0), k, x.size(2))), 1)
            .unfold(1, size=k, step=1)
            .permute(3, 0, 1, 2)
        )

    def _build_mask_and_subsample(
        self, not_dones, valids,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        t = not_dones.size(1)

        valid_mask = (not_dones > 0) & valids
        valid_mask = valid_mask.float()

        valid_mask_unfolded = self._build_unfolded(
            valid_mask[:, :-1].to(dtype=torch.bool), self.k
        )

        time_subsample = torch.randperm(
            t - 1, device=valid_mask.device, dtype=torch.long
        )[0:self.time_subsample]

        forward_mask = (
            torch.cumprod(valid_mask_unfolded.index_select(2, time_subsample), dim=0)
            .to(dtype=torch.bool)
            .flatten(1, 2)
        )

        max_k = forward_mask.flatten(1).any(-1).nonzero().max().item() + 1

        unroll_subsample = torch.randperm(max_k, dtype=torch.long)[0:self.forward_subsample]

        max_k = unroll_subsample.max().item() + 1

        unroll_subsample = unroll_subsample.to(device=valid_mask.device)
        forward_mask = forward_mask.index_select(0, unroll_subsample)

        return forward_mask, unroll_subsample, time_subsample, max_k

    def forward(self, actions, not_dones, valids, rnn_inputs, rnn_outputs):
        n = actions.size(0)
        t = actions.size(1)

        mask_res = self._build_mask_and_subsample(not_dones, valids)
        (forward_mask, unroll_subsample, time_subsample, max_k,) = mask_res

        actions = self.embed_actions(actions.long())
        actions_unfolded = self._build_unfolded(actions[:, :-1], max_k).index_select(
            2, time_subsample
        )

        rnn_outputs_subsampled = rnn_outputs[:, :-1].index_select(1, time_subsample)
        forward_preds, _ = self.rnn(
            actions_unfolded.contiguous().flatten(1, 2),
            rnn_outputs_subsampled.contiguous().view(1, -1, self.hidden_size),
        )
        forward_preds = forward_preds.index_select(0, unroll_subsample)
        forward_targets = self._build_unfolded(rnn_inputs[:, 1:], max_k)
        forward_targets = (
            forward_targets.index_select(2, time_subsample)
            .index_select(0, unroll_subsample)
            .flatten(1, 2)
        )

        positives = self.predictor(torch.cat((forward_preds, forward_targets), dim=-1))
        positive_loss = F.binary_cross_entropy_with_logits(
            positives, torch.broadcast_tensors(positives, positives.new_ones(()))[1], reduction='none'
        )
        positive_loss = torch.masked_select(positive_loss, forward_mask).mean()

        forward_negatives = torch.randint(
            0, n * t, size=(self.forward_subsample * self.time_subsample * n * 20,), dtype=torch.long, device=actions.device
        )
        forward_negatives = (
            rnn_inputs.flatten(0, 1)
            .index_select(0, forward_negatives)
            .view(self.forward_subsample, self.time_subsample * n, 20, -1)
        )
        negatives = self.predictor(
            torch.cat(
                (
                    forward_preds.view(self.forward_subsample, self.time_subsample * n, 1, -1)
                        .expand(-1, -1, 20, -1),
                    forward_negatives,
                ),
                dim=-1,
            )
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives, torch.broadcast_tensors(negatives, negatives.new_zeros(()))[1], reduction='none'
        )
        negative_loss = torch.masked_select(
            negative_loss, forward_mask.unsqueeze(2)
        ).mean()

        return 0.1 * (positive_loss + negative_loss)
