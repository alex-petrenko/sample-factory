from signal_slot.signal_slot import EventLoop, EventLoopObject, Timer, process_name, signal

from sample_factory.utils.utils import log


class HeartbeatStoppableEventLoopObject(EventLoopObject):
    def __init__(self, evt_loop: EventLoop, unique_name: str, interval_sec: int = 10):
        EventLoopObject.__init__(self, evt_loop, unique_name)
        self.heartbeat_timer = Timer(evt_loop, interval_sec)
        self.heartbeat_timer.timeout.connect(self._report_heartbeat)

    @signal
    def heartbeat(self):
        ...

    @signal
    def stop(self):
        ...

    def _report_heartbeat(self):
        p_name = process_name(self.event_loop.process)
        qsize = self.event_loop.signal_queue.qsize()
        self.heartbeat.emit(type(self), self.object_id, p_name, qsize)

    def on_stop(self, *_) -> None:
        """
        Default implementation, likely needs to be overridden in concrete classes to add
        termination logic.
        """
        log.debug(f"Stopping {self.object_id}...")

        if self.event_loop.owner is self:
            self.event_loop.stop()

        self.heartbeat_timer.stop()

        self.detach()  # remove from the current event loop
