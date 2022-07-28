from signal_slot.signal_slot import EventLoop, EventLoopObject, signal

from sample_factory.utils.utils import log


class StoppableEventLoopObject(EventLoopObject):
    def __init__(self, evt_loop: EventLoop, unique_name: str):
        EventLoopObject.__init__(self, evt_loop, unique_name)

    @signal
    def stop(self):
        ...

    def on_stop(self, *_) -> None:
        """
        Default implementation, likely needs to be overridden in concrete classes to add
        termination logic.
        """
        log.debug(f"Stopping {self.object_id}...")

        if self.event_loop.owner is self:
            self.event_loop.stop()

        self.detach()  # remove from the current event loop
