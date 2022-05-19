from typing import Iterable, Dict, Optional, List, Tuple

import torch

from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.signal_slot.signal_slot import signal, EventLoopObject, EventLoop

# TODO: this whole class should go
# We only really need two things:
# Async batcher that batches together and copies trajectories into training batches
# Sync batcher which reuses the same trajectories on sampler and learner in order to avoid copying (and I'm not sure we even need it)
# Sync batcher can really be just a circular buffer. And async batcher needs an entirely different logic anyway.
# WTF did I even write all of that
from sample_factory.utils.typing import PolicyID, MpQueue, Device
from sample_factory.utils.utils import AttrDict


class SliceMerger:
    def __init__(self):
        self.slice_starts: Dict[int, slice] = dict()
        self.slice_stops: Dict[int, slice] = dict()

    def _add_slice(self, s):
        self.slice_starts[s.start] = s
        self.slice_stops[s.stop] = s

    def _del_slice(self, s: slice):
        del self.slice_starts[s.start]
        del self.slice_stops[s.stop]

    def merge_slices(self, trajectory_slice: slice):
        new_slice = None

        if prev_slice := self.slice_stops.get(trajectory_slice.start):
            # merge with a slice that preceeds ours in the buffer
            new_slice = slice(prev_slice.start, trajectory_slice.stop)
            # delete the previous slice from both maps
            self._del_slice(prev_slice)
        elif next_slice := self.slice_starts.get(trajectory_slice.stop):
            # merge with a slice that is next in the buffer
            new_slice = slice(trajectory_slice.start, next_slice.stop)
            self._del_slice(next_slice)

        if new_slice:
            # successfully merged some slices, keep going
            self.merge_slices(new_slice)
        else:
            # nothing to merge, just add a new slice
            self._add_slice(trajectory_slice)

    def get_batch(self, batch_size) -> Optional[slice]:
        """
        At this point, all trajectory slices that share a boundary should have been merged into longer slices.
        If there's a slice that is at least trajectories_per_batch long starting where the previous returned slice
        ends - we found our batch.
        :return: a slice of trajectory buffer that will be a training batch on the learner
        """
        for slice_start, batch_slice in self.slice_starts.items():
            slice_len = batch_slice.stop - batch_slice.start
            if slice_len >= batch_size:
                self._del_slice(batch_slice)
                if slice_len > batch_size:
                    # we have even more trajectories than we need, keep them for later use
                    remaining_slice = slice(batch_slice.start + batch_size, batch_slice.stop)
                    self._add_slice(remaining_slice)
                    batch_slice = slice(batch_slice.start, batch_slice.start + batch_size)

                return batch_slice

        return None


class Batcher(EventLoopObject):
    def __init__(self, evt_loop: EventLoop, policy_id: PolicyID, unique_name):
        EventLoopObject.__init__(self, evt_loop, unique_name)
        self.policy_id = policy_id


class SequentialBatcher(Batcher):
    def __init__(self, evt_loop: EventLoop, policy_id: PolicyID, buffer_mgr, cfg: AttrDict):
        unique_name = f'{SequentialBatcher.__name__}_{policy_id}'
        Batcher.__init__(self, evt_loop, policy_id, unique_name)

        self.cfg = cfg
        self.policy_id = policy_id

        self.trajectories_per_training_batch = buffer_mgr.trajectories_per_batch
        self.trajectories_per_sampling_batch = buffer_mgr.worker_samples_per_iteration

        self.next_batch_start = 0

        self.training_slices: Dict[Device, SliceMerger] = dict()
        self.sampling_slices: Dict[Device, SliceMerger] = dict()
        for device in buffer_mgr.traj_tensors.keys():
            self.training_slices[device] = SliceMerger()
            self.sampling_slices[device] = SliceMerger()

        self.traj_buffer_queues = buffer_mgr.traj_buffer_queues
        self.traj_tensors = buffer_mgr.traj_tensors
        self.training_batches = buffer_mgr.training_batches[policy_id]
        self.available_batches = list(range(len(self.training_batches)))
        self.traj_tensors_to_release: List[Optional[Tuple[Device, slice]]] = [None] * len(self.available_batches)

    @signal
    def initialized(self): pass

    @signal
    def trajectory_buffers_available(self): pass

    @signal
    def training_batches_available(self): pass

    @signal
    def stop(self): pass

    def init(self):
        # there's nothing to do really
        self.initialized.emit()

    def on_new_trajectories(self, trajectory_dicts: Iterable[Dict], device: str):
        for trajectory_dict in trajectory_dicts:
            assert trajectory_dict['policy_id'] == self.policy_id
            trajectory_slice = trajectory_dict['traj_buffer_idx']
            assert isinstance(trajectory_slice, slice)  # TODO: support individual trajectories!
            self.training_slices[device].merge_slices(trajectory_slice)

        # TODO: if we're copying and in sync mode, we should remember to release the trajectories later when we are done training

        self._maybe_enqueue_new_training_batches(device)

    def _maybe_enqueue_new_training_batches(self, device: str):
        while self.available_batches:
            training_batch_slice = self.training_slices[device].get_batch(self.trajectories_per_training_batch)
            if training_batch_slice is None:
                break

            # obtain the index of the available batch buffer
            batch_idx = self.available_batches[-1]
            del self.available_batches[-1]

            # copy data into the training buffer
            self.training_batches[batch_idx][:] = self.traj_tensors[device][training_batch_slice]

            # signal the learner that we have a new training batch
            self.training_batches_available.emit(batch_idx)

            if self.cfg.async_rl:
                self._release_traj_tensors(device, training_batch_slice)
            else:
                self.traj_tensors_to_release[batch_idx] = (device, training_batch_slice)

        if not self.available_batches:
            # TODO: signal the inference worker to stop collecting experience - we accumulated enough training batches
            pass

    def on_training_batch_released(self, batch_idx: int):
        if self.traj_tensors_to_release[batch_idx] is not None:
            device, training_batch_slice = self.traj_tensors_to_release[batch_idx]
            self._release_traj_tensors(device, training_batch_slice)

        self.available_batches.append(batch_idx)

    def _release_traj_tensors(self, device: str, traj_slice: slice):
        self.sampling_slices[device].merge_slices(traj_slice)

        new_sampling_batches = []
        while (sampling_batch := self.sampling_slices[device].get_batch(self.trajectories_per_sampling_batch)) is not None:
            new_sampling_batches.append(sampling_batch)

        if new_sampling_batches:
            self.traj_buffer_queues[device].put_many(new_sampling_batches)
            self.trajectory_buffers_available.emit()

    def on_stop(self, emitter_id):
        self.stop.emit(self.object_id)
        if self.event_loop.owner is self:
            self.event_loop.stop()
        self.detach()  # remove from the current event loop
