import multiprocessing
from typing import List, Optional

from sample_factory.algo.inference.inference_worker import init_inference_process
from sample_factory.algo.learning.learner import init_learner_process
from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.sampling.rollout_worker import init_rollout_worker_process
from sample_factory.algo.utils.context import sf_global_context
from sample_factory.signal_slot.signal_slot import EventLoop, EventLoopProcess
from sample_factory.utils.utils import log


class ParallelRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.processes: List[EventLoopProcess] = []

    def multiprocessing_context(self) -> Optional[multiprocessing.context.BaseContext]:
        return multiprocessing.get_context("spawn")

    def init(self):
        super().init()

        for policy_id in range(self.cfg.num_policies):
            batcher_event_loop = EventLoop("batcher_evt_loop")
            self.batchers[policy_id] = self._make_batcher(batcher_event_loop, policy_id)
            batcher_event_loop.owner = self.batchers[policy_id]

            learner_proc = EventLoopProcess(f"learner_proc{policy_id}", self.mp_ctx, init_func=init_learner_process)
            self.processes.append(learner_proc)

            self.learners[policy_id] = self._make_learner(learner_proc.event_loop, policy_id, self.batchers[policy_id])
            learner_proc.event_loop.owner = self.learners[policy_id]
            learner_proc.set_init_func_args((sf_global_context(), self.learners[policy_id]))

            self.inference_workers[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                inference_proc = EventLoopProcess(
                    f"inference_proc{policy_id}-{i}", self.mp_ctx, init_func=init_inference_process
                )
                self.processes.append(inference_proc)
                inference_worker = self._make_inference_worker(
                    inference_proc.event_loop, policy_id, i, self.learners[policy_id].param_server
                )
                inference_proc.event_loop.owner = inference_worker
                inference_proc.set_init_func_args((sf_global_context(), inference_worker))
                self.inference_workers[policy_id].append(inference_worker)

        for i in range(self.cfg.num_workers):
            rollout_proc = EventLoopProcess(f"rollout_proc{i}", self.mp_ctx, init_func=init_rollout_worker_process)
            self.processes.append(rollout_proc)
            rollout_worker = self._make_rollout_worker(rollout_proc.event_loop, i)
            rollout_proc.event_loop.owner = rollout_worker
            rollout_proc.set_init_func_args((sf_global_context(), rollout_worker))
            self.rollout_workers.append(rollout_worker)

        self.connect_components()

    def _on_start(self):
        self._start_processes()

    def _start_processes(self):
        log.debug("Starting all processes...")
        for p in self.processes:
            log.debug(f"Starting process {p.name}")
            p.start()
            self.event_loop.process_events()

    def _on_everything_stopped(self):
        for p in self.processes:
            log.debug(f"Waiting for process {p.name} to stop...")
            p.join()

        super()._on_everything_stopped()
