from typing import Tuple

import torch

# noinspection PyPep8Naming
from torch.nn.utils.rnn import PackedSequence, invert_permutation

from sample_factory.utils.utils import log


def _build_pack_info_from_dones(
    dones: torch.Tensor, T: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create the indexing info needed to make the PackedSequence based on the dones.

    PackedSequences are PyTorch's way of supporting a single RNN forward
    call where each input in the batch can have an arbitrary sequence length

    They work as follows: Given the sequences [c], [x, y, z], [a, b],
    we generate data [x, a, c, y, b, z] and batch_sizes [3, 2, 1].  The
    data is a flattened out version of the input sequences (the ordering in
    data is determined by sequence length).  batch_sizes tells you that
    for each index, how many sequences have a length of (index + 1) or greater.
    This method will generate the new index ordering such that you can
    construct the data for a PackedSequence from a (N*T, ...) tensor
    via x.index_select(0, select_inds)
    """

    num_samples = len(dones)

    rollout_boundaries = dones.clone().detach()
    rollout_boundaries[T - 1 :: T] = 1  # end of each rollout is the boundary
    rollout_boundaries = rollout_boundaries.nonzero(as_tuple=False).squeeze(dim=1) + 1

    first_len = rollout_boundaries[0].unsqueeze(0)

    if len(rollout_boundaries) <= 1:
        log.debug(
            "Only one rollout boundary. This can happen if batch size is 1, probably not during the real training."
        )
        rollout_lengths = first_len
    else:
        rollout_lengths = rollout_boundaries[1:] - rollout_boundaries[:-1]
        rollout_lengths = torch.cat([first_len, rollout_lengths])

    rollout_starts_orig = rollout_boundaries - rollout_lengths

    # done=True for the last step in the episode, so done flags rolled 1 step to the right will indicate
    # first frames in the episodes
    is_new_episode = dones.clone().detach().view((-1, T))
    is_new_episode = is_new_episode.roll(1, 1)

    # roll() is cyclical, so done=True in the last position in the rollout will roll to 0th position
    # we want to avoid it here. (note to self: is there a function that does two of these things at once?)
    is_new_episode[:, 0] = 0
    is_new_episode = is_new_episode.view((-1,))

    lengths, sorted_indices = torch.sort(rollout_lengths, descending=True)
    # We will want these on the CPU for torch.unique_consecutive,
    # so move now.
    cpu_lengths = lengths.to(device="cpu", non_blocking=True)

    # We need to keep the original unpermuted rollout_starts, because the permutation is later applied
    # internally in the RNN implementation.
    # From modules/rnn.py:
    #       Each batch of the hidden state should match the input sequence that
    #       the user believes he/she is passing in.
    #       hx = self.permute_hidden(hx, sorted_indices)
    rollout_starts_sorted = rollout_starts_orig.index_select(0, sorted_indices)

    select_inds = torch.empty(num_samples, device=dones.device, dtype=torch.int64)

    max_length = int(cpu_lengths[0].item())
    # batch_sizes is *always* on the CPU
    batch_sizes = torch.empty((max_length,), device="cpu", dtype=torch.int64)

    offset = 0
    prev_len = 0
    num_valid_for_length = lengths.size(0)

    unique_lengths = torch.unique_consecutive(cpu_lengths)
    # Iterate over all unique lengths in reverse as they sorted
    # in decreasing order
    for i in range(len(unique_lengths) - 1, -1, -1):
        valids = lengths[0:num_valid_for_length] > prev_len
        num_valid_for_length = int(valids.float().sum().item())

        next_len = int(unique_lengths[i])

        batch_sizes[prev_len:next_len] = num_valid_for_length

        new_inds = (
            rollout_starts_sorted[0:num_valid_for_length].view(1, num_valid_for_length)
            + torch.arange(prev_len, next_len, device=rollout_starts_sorted.device).view(next_len - prev_len, 1)
        ).view(-1)

        # for a set of sequences [1, 2, 3], [4, 5], [6, 7], [8]
        # these indices will be 1,4,6,8,2,5,7,3
        # (all first steps in all trajectories, then all second steps, etc.)
        select_inds[offset : offset + new_inds.numel()] = new_inds

        offset += new_inds.numel()

        prev_len = next_len

    # Make sure we have an index for all elements
    assert offset == num_samples
    assert is_new_episode.shape[0] == num_samples

    return rollout_starts_orig, is_new_episode, select_inds, batch_sizes, sorted_indices


def build_rnn_inputs(x, dones_cpu, rnn_states, T: int):
    """
    Create a PackedSequence input for an RNN such that each
    set of steps that are part of the same episode are all part of
    a batch in the PackedSequence.
    Use the returned select_inds and build_core_out_from_seq to invert this.
    :param x: A (N*T, -1) tensor of the data to build the PackedSequence out of
    :param dones_cpu: A (N*T) tensor where dones[i] == 1.0 indicates an episode is done, a CPU-bound tensor
    :param rnn_states: A (N*T, -1) tensor of the rnn_hidden_states
    :param T: The length of the rollout
    :return: tuple(x_seq, rnn_states, select_inds)
        WHERE
        x_seq is the PackedSequence version of x to pass to the RNN
        rnn_states are the corresponding rnn state, zeroed on the episode boundary
        inverted_select_inds can be passed to build_core_out_from_seq so the RNN output can be retrieved
    """
    rollout_starts, is_new_episode, select_inds, batch_sizes, sorted_indices = _build_pack_info_from_dones(dones_cpu, T)
    inverted_select_inds = invert_permutation(select_inds)

    def device(t):
        return t.to(device=x.device)

    select_inds = device(select_inds)
    inverted_select_inds = device(inverted_select_inds)
    sorted_indices = device(sorted_indices)
    rollout_starts = device(rollout_starts)
    is_new_episode = device(is_new_episode)

    x_seq = PackedSequence(x.index_select(0, select_inds), batch_sizes, sorted_indices)

    # We zero-out rnn states for timesteps at the beginning of the episode.
    # rollout_starts are indices of all starts of sequences
    # (which can be due to episode boundary or just boundary of a rollout)
    # (1 - is_new_episode.view(-1, 1)).index_select(0, rollout_starts) gives us a zero for every beginning of
    # the sequence that is actually also a start of a new episode, and by multiplying this RNN state by zero
    # we ensure no information transfer across episode boundaries.
    rnn_states = rnn_states.index_select(0, rollout_starts)
    is_same_episode = (1 - is_new_episode.view(-1, 1)).index_select(0, rollout_starts)
    rnn_states = rnn_states * is_same_episode

    return x_seq, rnn_states, inverted_select_inds


def build_core_out_from_seq(x_seq: PackedSequence, inverted_select_inds):
    return x_seq.data.index_select(0, inverted_select_inds)
