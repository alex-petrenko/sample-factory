from collections import deque

from sample_factory.cfg.configurable import Configurable


class CommBroker:
    def send_msg(self, msg):
        raise NotImplementedError()

    def send_msgs(self, msgs):
        raise NotImplementedError()

    def get_msgs(self, block, timeout):
        raise NotImplementedError()


class SyncCommBroker(CommBroker):
    def __init__(self):
        super().__init__()

        # in synchronous learning scenario we don't want to waste time on unnecessary locking, hence this overload
        # exists that uses a regular unsynchronized queue
        self.msg_queue = deque([])

    def send_msg(self, msg):
        self.msg_queue.append(msg)

    def send_msgs(self, msgs):
        for msg in msgs:
            self.send_msg(msg)

    def get_msgs(self, block, timeout):
        msgs = list(self.msg_queue)
        self.msg_queue.clear()
        return msgs
