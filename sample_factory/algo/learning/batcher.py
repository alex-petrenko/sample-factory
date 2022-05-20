import random
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
from sample_factory.utils.utils import AttrDict, log


def slice_len(s: slice) -> int:
    return s.stop - s.start


class SliceMerger:
    def __init__(self):
        self.slice_starts: Dict[int, slice] = dict()
        self.slice_stops: Dict[int, slice] = dict()
        self.total_num = 0

    def _add_slice(self, s):
        self.slice_starts[s.start] = s
        self.slice_stops[s.stop] = s
        self.total_num += slice_len(s)

    def _del_slice(self, s: slice):
        del self.slice_starts[s.start]
        del self.slice_stops[s.stop]
        self.total_num -= slice_len(s)

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

    def _extract_at_most(self, s: slice, batch_size: int) -> slice:
        n = slice_len(s)
        self._del_slice(s)
        if n > batch_size:
            remaining_slice = slice(s.start + batch_size, s.stop)
            self._add_slice(remaining_slice)
            s = slice(s.start, s.start + batch_size)

        return s

    def get_at_most(self, batch_size) -> Optional[slice]:
        for s in self.slice_starts.values():
            return self._extract_at_most(s, batch_size)

        return None

    def get_exactly(self, batch_size: int) -> Optional[slice]:
        """
        At this point, all trajectory slices that share a boundary should have been merged into longer slices.
        If there's a slice that is at least trajectories_per_batch long starting where the previous returned slice
        ends - we found our batch.
        :return: a slice of trajectory buffer that will be a training batch on the learner
        """
        for slice_start, s in self.slice_starts.items():
            n = slice_len(s)
            if n >= batch_size:
                return self._extract_at_most(s, batch_size)

        return None


class Batcher(EventLoopObject):
    def __init__(self, evt_loop: EventLoop, policy_id: PolicyID, buffer_mgr, cfg: AttrDict):
        unique_name = f'{Batcher.__name__}_{policy_id}'
        super().__init__(evt_loop, unique_name)

        self.cfg = cfg
        self.policy_id = policy_id

        self.trajectories_per_training_batch = buffer_mgr.trajectories_per_batch
        self.trajectories_per_sampling_batch = buffer_mgr.worker_samples_per_iteration

        self.slices_for_training: Dict[Device, SliceMerger] = dict()
        self.slices_for_sampling: Dict[Device, SliceMerger] = dict()
        for device in buffer_mgr.traj_tensors.keys():
            self.slices_for_training[device] = SliceMerger()
            self.slices_for_sampling[device] = SliceMerger()

        self.traj_buffer_queues = buffer_mgr.traj_buffer_queues
        self.traj_tensors = buffer_mgr.traj_tensors
        self.training_batches = buffer_mgr.training_batches[policy_id]
        self.available_batches = list(range(len(self.training_batches)))
        self.traj_tensors_to_release: List[List[Tuple[Device, slice]]] = [[] for _ in range(len(self.available_batches))]

    @signal
    def initialized(self): pass

    @signal
    def trajectory_buffers_available(self): pass

    @signal
    def training_batches_available(self): pass

    @signal
    def stop_experience_collection(self): pass

    @signal
    def resume_experience_collection(self): pass

    @signal
    def stop(self): pass

    def init(self):
        # there's nothing to do really
        self.initialized.emit()

    def on_new_trajectories(self, trajectory_dicts: Iterable[Dict], device: str):
        for trajectory_dict in trajectory_dicts:
            assert trajectory_dict['policy_id'] == self.policy_id
            trajectory_slice = trajectory_dict['traj_buffer_idx']
            if not isinstance(trajectory_slice, slice):
                trajectory_slice = slice(trajectory_slice, trajectory_slice + 1)  # slice of len 1
            self.slices_for_training[device].merge_slices(trajectory_slice)

        self._maybe_enqueue_new_training_batches()

    def _maybe_enqueue_new_training_batches(self):
        with torch.no_grad():
            while self.available_batches:
                total_num_trajectories = 0
                for slices in self.slices_for_training.values():
                    total_num_trajectories += slices.total_num

                if total_num_trajectories < self.trajectories_per_training_batch:
                    # not enough experience yet to start training
                    break

                # obtain the index of the available batch buffer
                batch_idx = self.available_batches[-1]
                del self.available_batches[-1]
                assert len(self.traj_tensors_to_release[batch_idx]) == 0

                # extract slices of trajectories and copy them to the training batch
                devices = list(self.slices_for_training.keys())
                random.shuffle(devices)  # so that no sampling device is preferred

                trajectories_copied = 0
                remaining = self.trajectories_per_training_batch - trajectories_copied
                for device in devices:
                    traj_tensors = self.traj_tensors[device]
                    slices = self.slices_for_training[device]
                    while remaining > 0 and (traj_slice := slices.get_at_most(remaining)):
                        # copy data into the training buffer
                        start = trajectories_copied
                        stop = start + slice_len(traj_slice)
                        self.training_batches[batch_idx][start:stop] = traj_tensors[traj_slice]

                        # remember that we need to release these trajectories
                        self.traj_tensors_to_release[batch_idx].append((device, traj_slice))

                        trajectories_copied += slice_len(traj_slice)
                        remaining = self.trajectories_per_training_batch - trajectories_copied

                assert trajectories_copied == self.trajectories_per_training_batch and remaining == 0

                # signal the learner that we have a new training batch
                self.training_batches_available.emit(batch_idx)

                if self.cfg.async_rl:
                    self._release_traj_tensors(batch_idx)
                    if not self.available_batches:
                        log.debug('Signal inference workers to stop experience collection...')
                        self.stop_experience_collection.emit()

    def on_training_batch_released(self, batch_idx: int):
        self._release_traj_tensors(batch_idx)

        if not self.available_batches and self.cfg.async_rl:
            log.debug('Signal inference workers to resume experience collection...')
            self.resume_experience_collection.emit()

        self.available_batches.append(batch_idx)

    def _release_traj_tensors(self, batch_idx: int):
        new_sampling_batches = dict()

        if self.cfg.batched_sampling:
            for device, traj_slice in self.traj_tensors_to_release[batch_idx]:
                self.slices_for_sampling[device].merge_slices(traj_slice)

            for device, slices in self.slices_for_sampling.items():
                new_sampling_batches[device] = []
                while (sampling_batch := self.slices_for_sampling[device].get_exactly(self.trajectories_per_sampling_batch)) is not None:
                    new_sampling_batches[device].append(sampling_batch)
        else:
            for device, traj_slice in self.traj_tensors_to_release[batch_idx]:
                if device not in new_sampling_batches:
                    new_sampling_batches[device] = []

                for i in range(traj_slice.start, traj_slice.stop):
                    new_sampling_batches[device].append(i)

        self.traj_tensors_to_release[batch_idx] = []

        for device, batches in new_sampling_batches.items():
            self.traj_buffer_queues[device].put_many(batches)
        self.trajectory_buffers_available.emit()

    def on_stop(self, emitter_id):
        self.stop.emit(self.object_id)
        if self.event_loop.owner is self:
            self.event_loop.stop()
        self.detach()  # remove from the current event loop
