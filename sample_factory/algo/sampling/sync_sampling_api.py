from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Thread
from typing import Dict, Iterable, Optional

from sample_factory.algo.sampling.evaluation_sampling_api import SamplingLoop
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.algo.utils.tensor_dict import TensorDict, clone_tensordict
from sample_factory.utils.typing import Config, InitModelData, PolicyID, StatusCode
from sample_factory.utils.utils import log


class SyncSamplingAPI:
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        buffer_mgr: Optional[BufferMgr] = None,
        param_servers: Optional[Dict[PolicyID, ParameterServer]] = None,
    ):
        self.sampling_loop: SamplingLoop = SamplingLoop(cfg, env_info, print_episode_info=False)
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
