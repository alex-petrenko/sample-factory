import multiprocessing
from typing import Dict

from sample_factory.algo.learning.learner import init_learner_process
from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.context import sf_global_context
from sample_factory.signal_slot.signal_slot import EventLoop, EventLoopProcess
from sample_factory.utils.typing import PolicyID


class AsyncRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.learner_processes: Dict[PolicyID, EventLoopProcess] = dict()

    def init(self):
        super().init()
        ctx = multiprocessing.get_context('spawn')

        for policy_id in range(self.cfg.num_policies):
            learner_proc = EventLoopProcess(
                f'learner_proc{policy_id}', ctx, init_func=init_learner_process,
                args=(sf_global_context(), self.cfg, policy_id)
            )
            self.learner_processes[policy_id] = learner_proc
            self.learners[policy_id] = self._make_learner(learner_proc.event_loop, policy_id)

            # batcher_evt_loop = learner_evt_loop
            # if self.cfg.train_in_background_thread:
            #     batcher_evt_loop = EventLoop(f'Batcher_{policy_id}_EvtLoop')
            # self.batchers[policy_id] = self._make_batcher(batcher_evt_loop, policy_id)  # TODO: batcher should be a part of the learner

            self.inference_workers[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                inference_worker_evt_loop = EventLoop(f'InferenceWorker_p{policy_id}-w{i}_EvtLoop')
                inference_worker = self._make_inference_worker(inference_worker_evt_loop, policy_id, i, self.learners[policy_id].param_server)
                self.inference_workers[policy_id].append(inference_worker)

        for i in range(self.cfg.num_workers):
            rollout_worker_evt_loop = EventLoop(f'RolloutWorker_{i}_EvtLoop')
            rollout_worker = self._make_rollout_worker(rollout_worker_evt_loop, i)
            self.rollout_workers.append(rollout_worker)

        self.connect_components()

        for p in self.learner_processes.values():
            p.start()
