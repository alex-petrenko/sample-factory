from typing import List

from signal_slot.signal_slot import EventLoop, EventLoopProcess

from sample_factory.algo.learning.learner_worker import init_learner_process
from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.sampling.sampler import ParallelSampler
from sample_factory.algo.utils.context import sf_global_context
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
from sample_factory.utils.typing import StatusCode
from sample_factory.utils.utils import log
from functools import partial

class ParallelRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.processes: List[EventLoopProcess] = []

    def init(self) -> StatusCode:
        status = super().init()
        if status != ExperimentStatus.SUCCESS:
            return status

        mp_ctx = get_mp_ctx(self.cfg.serial_mode)

        for policy_id in range(self.cfg.num_policies):
            self.batchers[policy_id] = {}
            self.learners[policy_id] = {}
            for gpu_id in range(self.cfg.gpu_per_policy):
                batcher_event_loop = EventLoop("batcher_evt_loop")
                self.batchers[policy_id][gpu_id] = self._make_batcher(batcher_event_loop, policy_id, gpu_id)
                batcher_event_loop.owner = self.batchers[policy_id][gpu_id]
                learner_proc = EventLoopProcess(f"learner_proc{policy_id}-{gpu_id}", 
                                                mp_ctx, 
                                                init_func=partial(init_learner_process, size=self.cfg.gpu_per_policy, rank=gpu_id))
                self.processes.append(learner_proc)

                self.learners[policy_id][gpu_id] = self._make_learner(
                    learner_proc.event_loop,
                    policy_id,
                    gpu_id,
                )
                learner_proc.event_loop.owner = self.learners[policy_id][gpu_id]
                learner_proc.set_init_func_args((sf_global_context(), self.learners[policy_id][gpu_id]))

        self.samplers = [self._make_sampler(ParallelSampler, self.event_loop, gpu_id) for gpu_id in range(self.cfg.gpu_per_policy)]

        self.connect_components()
        return status

    def _on_start(self):
        self._start_processes()
        super()._on_start()

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

        for sampler in self.samplers:
            sampler.join()
        super()._on_everything_stopped()
