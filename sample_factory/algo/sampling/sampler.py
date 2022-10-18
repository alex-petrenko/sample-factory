from abc import ABC
from typing import Callable, Dict, List

from signal_slot.queue_utils import get_queue
from signal_slot.signal_slot import BoundMethod, EventLoop, EventLoopObject, EventLoopProcess, signal

from sample_factory.algo.sampling.inference_worker import InferenceWorker, init_inference_process
from sample_factory.algo.sampling.rollout_worker import RolloutWorker, init_rollout_worker_process
from sample_factory.algo.utils.context import sf_global_context
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.heartbeat import HeartbeatStoppableEventLoopObject
from sample_factory.algo.utils.misc import advance_rollouts_signal, new_trajectories_signal
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.typing import Config, MpQueue, PolicyID
from sample_factory.utils.utils import log


class AbstractSampler(EventLoopObject, Configurable):
    def __init__(
        self,
        event_loop: EventLoop,
        buffer_mgr: BufferMgr,
        param_servers: Dict[PolicyID, ParameterServer],
        cfg: Config,
        env_info: EnvInfo,
        unique_name: str,
    ):
        EventLoopObject.__init__(self, event_loop, object_id=unique_name)
        Configurable.__init__(self, cfg)

        self.buffer_mgr: BufferMgr = buffer_mgr
        self.policy_param_server: Dict[PolicyID, ParameterServer] = param_servers
        self.env_info: EnvInfo = env_info

    @signal
    def started(self):
        ...

    @signal
    def initialized(self):
        ...

    def init(self) -> None:
        raise NotImplementedError()

    def connect_model_initialized(self, policy_id: PolicyID, model_initialized_signal: signal) -> None:
        raise NotImplementedError()

    def connect_on_new_trajectories(self, policy_id: PolicyID, on_new_trajectories_handler: BoundMethod) -> None:
        raise NotImplementedError()

    def connect_trajectory_buffers_available(self, buffers_available_signal: signal) -> None:
        raise NotImplementedError()

    def connect_stop_experience_collection(self, stop_experience_collection_signal: signal) -> None:
        raise NotImplementedError()

    def connect_resume_experience_collection(self, resume_experience_collection_signal: signal) -> None:
        raise NotImplementedError()

    def connect_report_msg(self, report_msg_handler: Callable) -> None:
        raise NotImplementedError()

    def connect_update_training_info(self, update_training_info_signal: signal) -> None:
        raise NotImplementedError()

    def stoppable_components(self) -> List[HeartbeatStoppableEventLoopObject]:
        raise NotImplementedError()

    def heartbeat_components(self) -> List[HeartbeatStoppableEventLoopObject]:
        raise NotImplementedError()

    def join(self) -> None:
        """This is where we could join processes or threads if sampler starts any."""
        raise NotImplementedError()


class Sampler(AbstractSampler, ABC):
    def __init__(
        self,
        event_loop: EventLoop,
        buffer_mgr: BufferMgr,
        param_servers: Dict[PolicyID, ParameterServer],
        cfg: Config,
        env_info: EnvInfo,
    ):
        unique_name = Sampler.__name__
        AbstractSampler.__init__(self, event_loop, buffer_mgr, param_servers, cfg, env_info, unique_name)

        self.inference_queues: Dict[PolicyID, MpQueue] = {
            p: get_queue(cfg.serial_mode) for p in range(self.cfg.num_policies)
        }
        self.inference_workers: Dict[PolicyID, List[InferenceWorker]] = dict()
        self.rollout_workers: List[RolloutWorker] = []

    # internal signals used for communication with the workers, these are not a part of the interface
    @signal
    def _init_inference_workers(self):
        ...

    @signal
    def _inference_workers_initialized(self):
        ...

    def _make_inference_worker(self, event_loop, policy_id: PolicyID, worker_idx: int, param_server: ParameterServer):
        return InferenceWorker(
            event_loop,
            policy_id,
            worker_idx,
            self.buffer_mgr,
            param_server,
            self.inference_queues[policy_id],
            self.cfg,
            self.env_info,
        )

    def _make_rollout_worker(self, event_loop, worker_idx: int):
        return RolloutWorker(event_loop, worker_idx, self.buffer_mgr, self.inference_queues, self.cfg, self.env_info)

    def _for_each_inference_worker(self, func: Callable[[InferenceWorker], None]) -> None:
        for policy_id in range(self.cfg.num_policies):
            for inference_worker in self.inference_workers[policy_id]:
                func(inference_worker)

    def _for_each_rollout_worker(self, func: Callable[[RolloutWorker], None]) -> None:
        for rollout_worker in self.rollout_workers:
            func(rollout_worker)

    def _for_each_worker(self, func: Callable[[HeartbeatStoppableEventLoopObject], None]) -> None:
        self._for_each_inference_worker(func)
        self._for_each_rollout_worker(func)

    def _connect_internal_components(self):
        """Setup basic signal-slot internal connections specfic for the sampler."""
        self._for_each_inference_worker(lambda w: w.initialized.connect(self._inference_worker_ready))

        for rollout_worker_idx in range(self.cfg.num_workers):
            # once all learners and inference workers are initialized, we can initialize rollout workers
            rollout_worker = self.rollout_workers[rollout_worker_idx]
            self._inference_workers_initialized.connect(rollout_worker.init)

            # inference worker signals to advance rollouts when actions are ready
            for policy_id in range(self.cfg.num_policies):
                for inference_worker_idx in range(self.cfg.policy_workers_per_policy):
                    self.inference_workers[policy_id][inference_worker_idx].connect(
                        advance_rollouts_signal(rollout_worker_idx),
                        rollout_worker.advance_rollouts,
                    )

            # We also connect to our own advance_rollouts signal to avoid getting stuck when we have nothing
            # to send to the inference worker. This can happen if we have an entire trajectory of inactive agents.
            rollout_worker.connect(advance_rollouts_signal(rollout_worker_idx), rollout_worker.advance_rollouts)

    def connect_model_initialized(self, policy_id: PolicyID, model_initialized_signal: signal) -> None:
        for inference_worker in self.inference_workers[policy_id]:
            model_initialized_signal.connect(inference_worker.init)

    def connect_on_new_trajectories(self, policy_id: PolicyID, on_new_trajectories_handler: BoundMethod) -> None:
        signal_name = new_trajectories_signal(policy_id)
        self._for_each_rollout_worker(lambda w: w.connect(signal_name, on_new_trajectories_handler))

    def connect_trajectory_buffers_available(self, buffers_available_signal: signal) -> None:
        self._for_each_rollout_worker(lambda w: buffers_available_signal.connect(w.on_trajectory_buffers_available))

    def connect_stop_experience_collection(self, stop_collect_signal: signal) -> None:
        self._for_each_inference_worker(lambda w: stop_collect_signal.connect(w.should_stop_experience_collection))

    def connect_resume_experience_collection(self, resume_collect_signal: signal) -> None:
        self._for_each_inference_worker(lambda w: resume_collect_signal.connect(w.should_resume_experience_collection))

    def connect_report_msg(self, report_msg_handler: BoundMethod) -> None:
        self._for_each_inference_worker(lambda w: w.report_msg.connect(report_msg_handler))
        self._for_each_rollout_worker(lambda w: w.report_msg.connect(report_msg_handler))

    def connect_update_training_info(self, update_training_info: signal) -> None:
        self._for_each_rollout_worker(lambda w: update_training_info.connect(w.on_update_training_info))

    def _inference_worker_ready(self, policy_id: PolicyID, worker_idx: int):
        assert not self.inference_workers[policy_id][worker_idx].is_ready
        log.info(f"Inference worker {policy_id}-{worker_idx} is ready!")
        self.inference_workers[policy_id][worker_idx].is_ready = True

        # check if all workers for all policies are ready
        all_ready = True
        for policy_id in range(self.cfg.num_policies):
            all_ready &= all(w.is_ready for w in self.inference_workers[policy_id])

        if all_ready:
            log.info("All inference workers are ready! Signal rollout workers to start!")
            self._inference_workers_initialized.emit()

        # during initialization it serves no purpose to wait for all rollout workers to finish initialization,
        # instead we can just report that we are ready and rollout workers will start producing trajectories
        # as soon as all env.reset() calls are done
        self.initialized.emit()

    def stoppable_components(self) -> List[HeartbeatStoppableEventLoopObject]:
        stoppable = []
        self._for_each_worker(lambda w: stoppable.append(w))
        return stoppable

    def heartbeat_components(self) -> List[HeartbeatStoppableEventLoopObject]:
        heartbeat = []
        self._for_each_worker(lambda w: heartbeat.append(w))
        return heartbeat


class SerialSampler(Sampler):
    def __init__(
        self,
        event_loop: EventLoop,
        buffer_mgr,
        param_servers: Dict[PolicyID, ParameterServer],
        cfg: Config,
        env_info: EnvInfo,
    ):
        Sampler.__init__(self, event_loop, buffer_mgr, param_servers, cfg, env_info)

        for policy_id in range(self.cfg.num_policies):
            self.inference_workers[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                param_server = self.policy_param_server[policy_id]
                inference_worker = self._make_inference_worker(self.event_loop, policy_id, i, param_server)
                self.inference_workers[policy_id].append(inference_worker)

        for i in range(self.cfg.num_workers):
            rollout_worker = self._make_rollout_worker(self.event_loop, i)
            self.rollout_workers.append(rollout_worker)

        self._connect_internal_components()

    def init(self) -> None:
        self.started.emit()

    def join(self) -> None:
        pass


class ParallelSampler(Sampler):
    def __init__(
        self,
        event_loop: EventLoop,
        buffer_mgr,
        param_servers: Dict[PolicyID, ParameterServer],
        cfg: Config,
        env_info: EnvInfo,
    ):
        Sampler.__init__(self, event_loop, buffer_mgr, param_servers, cfg, env_info)
        self.processes: List[EventLoopProcess] = []
        mp_ctx = get_mp_ctx(cfg.serial_mode)

        for policy_id in range(self.cfg.num_policies):
            self.inference_workers[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                inference_proc = EventLoopProcess(
                    f"inference_proc{policy_id}-{i}", mp_ctx, init_func=init_inference_process
                )
                self.processes.append(inference_proc)
                inference_worker = self._make_inference_worker(
                    inference_proc.event_loop,
                    policy_id,
                    i,
                    self.policy_param_server[policy_id],
                )
                inference_proc.event_loop.owner = inference_worker
                inference_proc.set_init_func_args((sf_global_context(), inference_worker))
                self.inference_workers[policy_id].append(inference_worker)

        for i in range(self.cfg.num_workers):
            rollout_proc = EventLoopProcess(f"rollout_proc{i}", mp_ctx, init_func=init_rollout_worker_process)
            self.processes.append(rollout_proc)
            rollout_worker = self._make_rollout_worker(rollout_proc.event_loop, i)
            rollout_proc.event_loop.owner = rollout_worker
            rollout_proc.set_init_func_args((sf_global_context(), rollout_worker))
            self.rollout_workers.append(rollout_worker)

        self._connect_internal_components()

    def init(self) -> None:
        log.debug("Starting all processes...")

        def start_process(p):
            log.debug(f"Starting process {p.name}")
            p.start()

        pool_size = min(16, len(self.processes))
        from multiprocessing.pool import ThreadPool

        with ThreadPool(pool_size) as pool:
            pool.map(start_process, self.processes)

        self.started.emit()

    def join(self) -> None:
        for p in self.processes:
            log.debug(f"Waiting for process {p.name} to join...")
            p.join()
