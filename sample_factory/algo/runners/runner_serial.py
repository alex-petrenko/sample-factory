import multiprocessing
from typing import Optional

from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.torch_utils import init_torch_runtime


class SerialRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trajectories_per_batch = self.cfg.batch_size // self.cfg.rollout

    def multiprocessing_context(self) -> Optional[multiprocessing.context.BaseContext]:
        return None

    def init(self):
        # in serial mode everything will be happening in the main process, so we need to initialize cuda
        init_torch_runtime(self.cfg, max_num_threads=None)
        super().init()

        for policy_id in range(self.cfg.num_policies):
            self.batchers[policy_id] = self._make_batcher(self.event_loop, policy_id)
            self.learners[policy_id] = self._make_learner(self.event_loop, policy_id, self.batchers[policy_id])

            self.inference_workers[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                inference_worker = self._make_inference_worker(self.event_loop, policy_id, i, self.learners[policy_id].param_server)
                self.inference_workers[policy_id].append(inference_worker)

        for i in range(self.cfg.num_workers):
            rollout_worker = self._make_rollout_worker(self.event_loop, i)
            self.rollout_workers.append(rollout_worker)

        self.connect_components()
