from sample_factory.algo.inference_worker import InferenceWorker
from sample_factory.algo.runners.runner import Runner
from sample_factory.signal_slot.signal_slot import EventLoop


class AsyncRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init(self):
        super().init()

        for policy_id in range(self.cfg.num_policies):
            learner_evt_loop = EventLoop(f'Learner_{policy_id}_EvtLoop')
            self.learners[policy_id] = self._make_learner(learner_evt_loop, policy_id)

            batcher_evt_loop = learner_evt_loop
            if self.cfg.train_in_background_thread:
                batcher_evt_loop = EventLoop(f'Batcher_{policy_id}_EvtLoop')
            self.batchers[policy_id] = self._make_batcher(batcher_evt_loop, policy_id)

            self.inference_workers[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                inference_worker_evt_loop = EventLoop(f'InferenceWorker_p{policy_id}-w{i}_EvtLoop')
                inference_worker = self._make_inference_worker(inference_worker_evt_loop, policy_id, i)
                self.inference_workers[policy_id].append(inference_worker)

        for i in range(self.cfg.num_workers):
            rollout_worker_evt_loop = EventLoop(f'RolloutWorker_{i}_EvtLoop')
            rollout_worker = self._make_rollout_worker(rollout_worker_evt_loop, i)
            self.rollout_workers.append(rollout_worker)

        # TODO: connect components