from sample_factory.algo.utils.queues import get_mp_queue
from sample_factory.cfg.configurable import Configurable


class NoopSampler(Sampler):
    def __init__(self, cfg,):
        super().__init__(cfg)

        self.experience_queue = get_mp_queue()

    def init(self):
        pass

    def get_trajectories(self, block, timeout):
        return self.experience_queue.get_many(block, timeout)

    def step(self):
        pass
