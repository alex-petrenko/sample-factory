from sample_factory.algo.runners.runner import Runner


class SyncRunner(Runner):
    def __init__(self, cfg, comm_broker, sampler, batcher, learner):
        super().__init__(cfg, comm_broker, sampler, batcher, learner)

        self.trajectories_per_batch = self.cfg.batch_size // self.cfg.rollout

    def algo_step(self, timing):
        with timing.add_time('sampling'):
            trajectories = self.sampler.get_trajectories_sync(timing)
        with timing.add_time('batching'):
            self.batcher.batch_trajectories(trajectories)

        with timing.add_time('training'):
            while (experience_batch := self.batcher.get_batch_sync()) is not None:
                self.learner.train_sync(experience_batch, timing)
