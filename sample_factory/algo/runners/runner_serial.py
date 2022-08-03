from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.sampling.sampler import SerialSampler
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.torch_utils import init_torch_runtime
from sample_factory.utils.typing import StatusCode


class SerialRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trajectories_per_batch = self.cfg.batch_size // self.cfg.rollout

    def init(self) -> StatusCode:
        # in serial mode everything will be happening in the main process, so we need to initialize cuda
        init_torch_runtime(self.cfg, max_num_threads=None)
        status = super().init()
        if status != ExperimentStatus.SUCCESS:
            return status

        for policy_id in range(self.cfg.num_policies):
            self.batchers[policy_id] = self._make_batcher(self.event_loop, policy_id)
            self.learners[policy_id] = self._make_learner(self.event_loop, policy_id, self.batchers[policy_id])

        self.sampler = self._make_sampler(SerialSampler, self.event_loop)

        self.connect_components()
        return status
