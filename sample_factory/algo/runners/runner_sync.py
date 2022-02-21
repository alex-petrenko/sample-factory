from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.communication_broker import SyncCommBroker


class SyncRunner(Runner):
    def __init__(self, cfg, comm_broker, sampler, batcher, learner):
        super().__init__(cfg, comm_broker, sampler, batcher, learner)

        self.trajectories_per_batch = self.cfg.batch_size // self.cfg.rollout

    def algo_step(self, timing):
        trajectories = self.sampler.get_trajectories_sync()

        self.batcher.batch_trajectories(trajectories)

        experience_batch = self.batcher.get_batch_sync()
        if experience_batch is not None:
            self.learner.train_sync(experience_batch, timing)
