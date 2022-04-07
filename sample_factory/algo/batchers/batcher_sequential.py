from typing import Iterable, Dict, Optional

from sample_factory.signal_slot.signal_slot import signal, EventLoopObject, EventLoop


class SequentialBatcher(EventLoopObject):
    def __init__(self, evt_loop: EventLoop, trajectories_per_batch: int, total_num_trajectories: int):
        unique_name = f'{SequentialBatcher.__name__}'
        EventLoopObject.__init__(self, evt_loop, unique_name)

        self.slice_starts: Dict[int, slice] = dict()
        self.slice_stops: Dict[int, slice] = dict()

        self.trajectories_per_batch = trajectories_per_batch
        self.total_num_trajectories = total_num_trajectories
        self.next_batch_start = 0

        self.event_loop.start.connect(self.init)

    @signal
    def new_batches(self): pass

    def init(self):
        pass

    def _add_slice(self, s):
        self.slice_starts[s.start] = s
        self.slice_stops[s.stop] = s

    def _del_slice(self, s: slice):
        del self.slice_starts[s.start]
        del self.slice_stops[s.stop]

    def _merge_slices(self, trajectory_slice: slice):
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
            self._merge_slices(new_slice)
        else:
            # nothing to merge, just add a new slice
            self._add_slice(trajectory_slice)

    def batch_trajectories(self, trajectory_slices: Iterable[slice]):
        for trajectory_slice in trajectory_slices:
            self._merge_slices(trajectory_slice)

    def get_batch_sync(self) -> Optional[slice]:
        """
        At this point, all trajectory slices that share a boundary should have been merged into longer slices.
        If there's a slice that is at least trajectories_per_batch long starting where the previous returned slice
        ends - we found our batch.
        :return: a slice of trajectory buffer that will be a training batch on the learner
        """
        if batch_slice := self.slice_starts.get(self.next_batch_start):
            slice_len = batch_slice.stop - batch_slice.start
            if slice_len >= self.trajectories_per_batch:
                self._del_slice(batch_slice)
                if slice_len > self.trajectories_per_batch:
                    # we have even more trajectories than we need, keep them for later use
                    remaining_slice = slice(batch_slice.start + self.trajectories_per_batch, batch_slice.stop)
                    self._add_slice(remaining_slice)
                    batch_slice = slice(batch_slice.start, batch_slice.start + self.trajectories_per_batch)

                self.next_batch_start = batch_slice.stop % self.total_num_trajectories
                assert self.total_num_trajectories - self.next_batch_start >= self.trajectories_per_batch, \
                    'A whole number of batches should be allocated. Logic error.'

                return batch_slice

        return None

    def on_new_trajectories(self, trajectory_slices: Iterable[slice]):
        self.batch_trajectories(trajectory_slices)

        new_batches = []
        while (experience_batch := self.get_batch_sync()) is not None:
            new_batches.append(experience_batch)

        if new_batches:
            self.new_batches.emit(new_batches)
