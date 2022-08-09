from __future__ import annotations

import time
from queue import Empty, Full, Queue
from threading import Thread
from typing import Callable, Dict, Iterable, Optional

from signal_slot.signal_slot import EventLoop, EventLoopObject, EventLoopStatus, signal

from sample_factory.algo.sampling.sampler import AbstractSampler, ParallelSampler, SerialSampler
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.algo.utils.tensor_dict import TensorDict, clone_tensordict
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.gpu_utils import set_global_cuda_envvars
from sample_factory.utils.typing import Config, InitModelData, PolicyID, StatusCode
from sample_factory.utils.utils import log


class SamplingLoop(EventLoopObject, Configurable):
    def __init__(self, cfg: Config, env_info: EnvInfo):
        Configurable.__init__(self, cfg)
        unique_name = SamplingLoop.__name__
        self.event_loop: EventLoop = EventLoop(unique_loop_name=f"{unique_name}_EvtLoop", serial_mode=cfg.serial_mode)
        self.event_loop.owner = self
        EventLoopObject.__init__(self, self.event_loop, object_id=unique_name)
        # self.event_loop.verbose = True

        self.env_info = env_info
        self.iteration: int = 0

        self.buffer_mgr: Optional[BufferMgr] = None
        self.param_servers: Optional[Dict[PolicyID, ParameterServer]] = None

        self.new_trajectory_callback: Optional[Callable] = None
        self.status: Optional[StatusCode] = None

        self.ready: bool = False
        self.stopped: bool = False

    @signal
    def model_initialized(self):
        ...

    @signal
    def trajectory_buffers_available(self):
        ...

    @signal
    def stop(self):
        ...

    def init(
        self, buffer_mgr: Optional[BufferMgr] = None, param_servers: Optional[Dict[PolicyID, ParameterServer]] = None
    ):
        set_global_cuda_envvars(self.cfg)

        self.buffer_mgr = buffer_mgr
        if self.buffer_mgr is None:
            self.buffer_mgr = BufferMgr(self.cfg, self.env_info)

        self.param_servers = param_servers
        if self.param_servers is None:
            self.param_servers = dict()
            for policy_id in range(self.cfg.num_policies):
                self.param_servers[policy_id] = ParameterServer(
                    policy_id, self.buffer_mgr.policy_versions, self.cfg.serial_mode
                )

        sampler_cls = SerialSampler if self.cfg.serial_mode else ParallelSampler
        sampler: AbstractSampler = sampler_cls(
            self.event_loop, self.buffer_mgr, self.param_servers, self.cfg, self.env_info
        )
        self.event_loop.start.connect(sampler.init)
        sampler.started.connect(self.on_sampler_started)
        sampler.initialized.connect(self.on_sampler_initialized)

        for policy_id in range(self.cfg.num_policies):
            sampler.connect_model_initialized(policy_id, self.model_initialized)
            sampler.connect_on_new_trajectories(policy_id, self.on_new_trajectories)
        sampler.connect_trajectory_buffers_available(self.trajectory_buffers_available)

        for stoppable in sampler.stoppable_components():
            self.stop.connect(stoppable.on_stop)

    def wait_until_ready(self):
        while not self.ready:
            log.debug(f"{self.object_id}: waiting for sampler to be ready...")
            time.sleep(0.5)

    def start(self, init_model_data: Optional[Dict[PolicyID, InitModelData]] = None):
        """Model initialization should kickstart the sampling loop."""
        for policy_id in range(self.cfg.num_policies):
            if init_model_data is None:
                self.model_initialized.emit(None)
            else:
                self.model_initialized.emit(init_model_data[policy_id])

    def set_new_trajectory_callback(self, cb: Callable) -> None:
        self.new_trajectory_callback = cb

    def on_sampler_started(self):
        self.ready = True

    def on_sampler_initialized(self):
        log.debug(f"{self.object_id}: sampler fully initialized!")

    def on_new_trajectories(self, trajectory_dicts: Iterable[Dict], device: str):
        for trajectory_dict in trajectory_dicts:
            traj_buffer_idx: int | slice = trajectory_dict["traj_buffer_idx"]
            if isinstance(traj_buffer_idx, slice):
                trajectory_slice = traj_buffer_idx
            else:
                trajectory_slice = slice(traj_buffer_idx, traj_buffer_idx + 1)

            # data for this trajectory is now available in the buffer
            # always use a slice so that returned tensors are the same dimensionality regardless of whether we
            # use batched or non-batched sampling
            traj = self.buffer_mgr.traj_tensors_torch[device][trajectory_slice]
            self.new_trajectory_callback(traj, [traj_buffer_idx], device)

    def yield_trajectory_buffers(self, available_buffers: Iterable[int | slice], device: str):
        # make this trajectory buffer available again
        self.buffer_mgr.traj_buffer_queues[device].put_many(available_buffers)
        self.iteration += 1
        for policy_id in range(self.cfg.num_policies):
            self.trajectory_buffers_available.emit(policy_id, self.iteration)

    def run(self) -> StatusCode:
        log.debug("Before event loop...")

        # noinspection PyBroadException
        try:
            evt_loop_status = self.event_loop.exec()
            self.status = (
                ExperimentStatus.INTERRUPTED
                if evt_loop_status == EventLoopStatus.INTERRUPTED
                else ExperimentStatus.SUCCESS
            )
            self.stop.emit()
        except Exception:
            log.exception(f"Uncaught exception in {self.object_id} evt loop")
            self.status = ExperimentStatus.FAILURE

        log.debug(f"{SamplingLoop.__name__} finished with {self.status=}")
        return self.status

    def stop_sampling(self):
        self.stop.emit()
        self.event_loop.stop()
        self.stopped = True


class SyncSamplingAPI:
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        buffer_mgr: Optional[BufferMgr] = None,
        param_servers: Optional[Dict[PolicyID, ParameterServer]] = None,
    ):
        self.sampling_loop: SamplingLoop = SamplingLoop(cfg, env_info)
        self.sampling_loop.init(buffer_mgr, param_servers)
        self.sampling_loop.set_new_trajectory_callback(self._on_new_trajectories)
        self.sampling_thread: Thread = Thread(target=self.sampling_loop.run)
        self.sampling_thread.start()

        self.sampling_loop.wait_until_ready()
        self.traj_queue: Queue = Queue(maxsize=100)

    def start(self, init_model_data: Optional[Dict[PolicyID, InitModelData]] = None):
        self.sampling_loop.start(init_model_data)

    def _on_new_trajectories(self, traj: TensorDict, traj_buffer_indices: Iterable[int | slice], device: str):
        traj_clone = clone_tensordict(traj)  # we copied the data so we can release the buffer

        # just release buffers after every trajectory
        # we could alternatively have more sophisticated logic here, see i.e. batcher.py
        self.sampling_loop.yield_trajectory_buffers(traj_buffer_indices, device)

        while not self.sampling_loop.stopped:
            try:
                self.traj_queue.put(traj_clone, timeout=1.0, block=True)
                break
            except Full:
                log.debug(f"{self._on_new_trajectories.__name__}: trajectory queue full, waiting...")
                self.sampling_loop.event_loop.process_events()

    def get_trajectories_sync(self) -> Optional[TensorDict]:
        while not self.sampling_loop.stopped:
            try:
                traj = self.traj_queue.get(timeout=5.0)
                return traj
            except Empty:
                log.debug(f"{self.get_trajectories_sync.__name__}(): waiting for trajectories...")
                continue

        return None

    def stop(self) -> StatusCode:
        self.sampling_loop.stop_sampling()
        self.sampling_thread.join()
        return self.sampling_loop.status
