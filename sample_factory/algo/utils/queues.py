import os


def get_mp_queue(buffer_size_bytes=1_000_000):
    if os.name == 'nt':
        from sample_factory.utils.faster_fifo_stub import MpQueueWrapper as MpQueue
    else:
        from faster_fifo import Queue as MpQueue
        # noinspection PyUnresolvedReferences
        import faster_fifo_reduction

    return MpQueue(buffer_size_bytes)
