import sys
import time
from typing import Dict, Iterable, Optional

from signal_slot.signal_slot import EventLoop, EventLoopObject, EventLoopStatus, StatusCode, Timer, signal

from sample_factory.algo.sampling.sampler import AbstractSampler, ParallelSampler
from sample_factory.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.gpu_utils import set_global_cuda_envvars
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.utils.utils import log
from sf_examples.atari_examples.train_atari import parse_atari_args, register_atari_components


class SamplingLoop(EventLoopObject, Configurable):
    def __init__(self, cfg: Config, env_info: EnvInfo, sample_env_steps: int):
        Configurable.__init__(self, cfg)
        unique_name = SamplingLoop.__name__
        self.event_loop: EventLoop = EventLoop(unique_loop_name=f"{unique_name}_EvtLoop", serial_mode=cfg.serial_mode)
        self.event_loop.owner = self
        EventLoopObject.__init__(self, self.event_loop, object_id=unique_name)

        self.env_info = env_info
        self.sample_env_steps = sample_env_steps
        self.samples_collected: int = 0
        self.prev_samples_collected: int = 0
        self.fps_tstamp = time.time()

        self.buffer_mgr: Optional[BufferMgr] = None

        Timer(self.event_loop, 1).timeout.connect(self.timer_callback)

    @signal
    def model_initialized(self):
        ...

    @signal
    def trajectory_buffers_available(self):
        ...

    @signal
    def stop(self):
        ...

    def init(self):
        set_global_cuda_envvars(self.cfg)
        self.buffer_mgr = BufferMgr(self.cfg, self.env_info)

        param_servers: Dict[PolicyID, ParameterServer] = dict()
        for policy_id in range(self.cfg.num_policies):
            param_servers[policy_id] = ParameterServer(
                policy_id, self.buffer_mgr.policy_versions, self.cfg.serial_mode, get_mp_ctx(self.cfg.serial_mode)
            )

        assert (
            not self.cfg.serial_mode
        ), "In serial mode the model has to be initialized in the main process (we can't leave the parameter server uninitialized)"

        sampler: AbstractSampler = ParallelSampler(
            self.event_loop, self.buffer_mgr, param_servers, self.cfg, self.env_info
        )
        self.event_loop.start.connect(sampler.init)

        for policy_id in range(self.cfg.num_policies):
            sampler.connect_model_initialized(policy_id, self.model_initialized)
            sampler.connect_on_new_trajectories(policy_id, self.on_new_trajectories)
        sampler.connect_trajectory_buffers_available(self.trajectory_buffers_available)

        for stoppable in sampler.stoppable_components():
            self.stop.connect(stoppable.on_stop)

        # This is a fake signal, we don't actually initialize the model in this example, just sampling
        # actions from a policy with random weights. Normally learners use this signal to broadcase their
        # weights to the sampler.
        self.model_initialized.emit(None)

    def on_new_trajectories(self, trajectory_dicts: Iterable[Dict], device: str):
        available_buffers = []

        for trajectory_dict in trajectory_dicts:
            traj_buffer_idx = trajectory_dict["traj_buffer_idx"]
            # data for this trajectory is now available in the buffer
            traj = self.buffer_mgr.traj_tensors_torch[device][traj_buffer_idx]
            self.samples_collected += len(traj["rewards"])
            available_buffers.append(traj_buffer_idx)

        # make this trajectory buffer available again
        self.buffer_mgr.traj_buffer_queues[device].put_many(available_buffers)
        self.trajectory_buffers_available.emit()

    def timer_callback(self):
        fps = (self.samples_collected - self.prev_samples_collected) / (time.time() - self.fps_tstamp)
        fps_frameskip = fps * self.cfg.env_frameskip

        self.prev_samples_collected = self.samples_collected
        self.fps_tstamp = time.time()

        fps_frameskip_str = f" ({fps_frameskip:.1f} FPS with frameskip)" if self.cfg.env_frameskip > 1 else ""
        log.debug(f"Samples collected: {self.samples_collected}, throughput: {fps:.1f} FPS{fps_frameskip_str}")
        if self.samples_collected >= self.sample_env_steps:
            self.stop.emit()
            self.event_loop.stop()

    def run(self) -> StatusCode:
        status = ExperimentStatus.SUCCESS

        # noinspection PyBroadException
        try:
            evt_loop_status = self.event_loop.exec()
            status = ExperimentStatus.INTERRUPTED if evt_loop_status == EventLoopStatus.INTERRUPTED else status
            self.stop.emit()
        except Exception:
            log.exception(f"Uncaught exception in {self.object_id} evt loop")
            status = ExperimentStatus.FAILURE

        log.debug(f"{SamplingLoop.__name__} finished with {status=}")
        return status


def sample(cfg: Config, env_info: EnvInfo, sample_env_steps: int = 1_000_000) -> StatusCode:
    sampling_loop = SamplingLoop(cfg, env_info, sample_env_steps)
    sampling_loop.init()
    return sampling_loop.run()


def main() -> StatusCode:
    register_atari_components()
    cfg = parse_atari_args()
    env_info = obtain_env_info_in_a_separate_process(cfg)
    return sample(cfg, env_info)


if __name__ == "__main__":
    sys.exit(main())
