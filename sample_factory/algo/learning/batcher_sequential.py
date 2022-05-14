from typing import Iterable, Dict, Optional

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.signal_slot.signal_slot import signal, EventLoopObject, EventLoop

# TODO: this whole class should go
# We only really need two things:
# Async batcher that batches together and copies trajectories into training batches
# Sync batcher which reuses the same trajectories on sampler and learner in order to avoid copying (and I'm not sure we even need it)
# Sync batcher can really be just a circular buffer. And async batcher needs an entirely different logic anyway.
# WTF did I even write all of that
from sample_factory.utils.typing import PolicyID, MpQueue


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
    def __init__(
            self, evt_loop: EventLoop, trajectories_per_batch: int, total_num_trajectories: int,
            env_info: EnvInfo, policy_id: PolicyID,
            traj_buffer_queue: MpQueue,
    ):
        unique_name = f'{SequentialBatcher.__name__}_{policy_id}'
        Batcher.__init__(self, evt_loop, policy_id, unique_name)

        self.policy_id = policy_id

        self.trajectories_per_training_batch = trajectories_per_batch
        self.trajectories_per_sampling_batch = env_info.num_agents  # TODO: this logic should be changed
        self.total_num_trajectories = total_num_trajectories
        self.next_batch_start = 0

        self.training_batches = SliceMerger()
        self.sampling_batches = SliceMerger()

        self.traj_buffer_queue = traj_buffer_queue

    @signal
    def initialized(self): pass

    @signal
    def trajectory_buffers_available(self): pass

    @signal
    def training_batches_available(self): pass

    @signal
    def stop(self): pass

    def init(self):
        # we should put all initial batches into the sampler queue
        for i in range(self.total_num_trajectories):
            self.on_training_batch_released(slice(i, i + 1))

        self.initialized.emit()

    def on_new_trajectories(self, trajectory_dicts: Iterable[Dict]):
        for trajectory_dict in trajectory_dicts:
            assert trajectory_dict['policy_id'] == self.policy_id
            trajectory_slice = trajectory_dict['traj_buffer_idx']
            self.training_batches.merge_slices(trajectory_slice)

        new_training_batches = []
        while (training_batch := self.training_batches.get_batch(self.trajectories_per_training_batch)) is not None:
            new_training_batches.append(training_batch)

        if new_training_batches:
            self.training_batches_available.emit(new_training_batches)

    def on_training_batch_released(self, batch: slice):
        self.sampling_batches.merge_slices(batch)
        new_sampling_batches = []
        while (sampling_batch := self.sampling_batches.get_batch(self.trajectories_per_sampling_batch)) is not None:
            new_sampling_batches.append(sampling_batch)

        if new_sampling_batches:
            self.traj_buffer_queue.put_many(new_sampling_batches)
            self.trajectory_buffers_available.emit()

    def on_stop(self, emitter_id):
        self.stop.emit(self.object_id)
        if self.event_loop.owner is self:
            self.event_loop.stop()
        self.detach()  # remove from the current event loop
