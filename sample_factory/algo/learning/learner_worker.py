from __future__ import annotations

import os
from threading import Thread
from typing import Dict, Optional

import psutil
import torch
from signal_slot.signal_slot import EventLoop, Timer, signal
from torch import Tensor

from sample_factory.algo.learning.batcher import Batcher
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.context import SampleFactoryContext, set_global_context
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.heartbeat import HeartbeatStoppableEventLoopObject
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.algo.utils.torch_utils import init_torch_runtime
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.gpu_utils import cuda_envvars_for_policy
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.utils.utils import init_file_logger, log


def init_learner_process(sf_context: SampleFactoryContext, learner_worker: LearnerWorker):
    set_global_context(sf_context)
    log.info(f"{learner_worker.object_id}\tpid {os.getpid()}\tparent {os.getppid()}")

    # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
    import signal as os_signal

    os_signal.signal(os_signal.SIGINT, os_signal.SIG_IGN)

    cfg = learner_worker.cfg
    init_file_logger(cfg)

    try:
        psutil.Process().nice(cfg.default_niceness)
    except psutil.AccessDenied:
        log.error("Low niceness requires sudo!")

    if cfg.device == "gpu":
        cuda_envvars_for_policy(learner_worker.learner.policy_id, "learning")

    init_torch_runtime(cfg)


class LearnerWorker(HeartbeatStoppableEventLoopObject, Configurable):
    def __init__(
        self,
        evt_loop: EventLoop,
        cfg: Config,
        env_info: EnvInfo,
        buffer_mgr: BufferMgr,
        batcher: Batcher,
        policy_id: PolicyID,
    ):
        Configurable.__init__(self, cfg)

        unique_name = f"{LearnerWorker.__name__}_p{policy_id}"
        HeartbeatStoppableEventLoopObject.__init__(self, evt_loop, unique_name, cfg.heartbeat_interval)

        self.batcher: Batcher = batcher
        self.batcher_thread: Optional[Thread] = None

        policy_versions_tensor: Tensor = buffer_mgr.policy_versions
        self.param_server = ParameterServer(policy_id, policy_versions_tensor, cfg.serial_mode)
        self.learner: Learner = Learner(cfg, env_info, policy_versions_tensor, policy_id, self.param_server)

        # total number of full training iterations (potentially multiple minibatches/epochs per iteration)
        self.training_iteration_since_resume: int = 0

        self.cache_cleanup_timer = Timer(self.event_loop, 30)
        self.cache_cleanup_timer.timeout.connect(self._cleanup_cache)

    @signal
    def initialized(self):
        ...

    @signal
    def model_initialized(self):
        ...

    @signal
    def report_msg(self):
        ...

    @signal
    def training_batch_released(self):
        ...

    @signal
    def finished_training_iteration(self):
        ...

    @signal
    def saved_model(self):
        ...

    @signal
    def stop(self):
        ...

    def save(self) -> bool:
        if self.learner.save():
            self.saved_model.emit(self.learner.policy_id)
            return True
        return False

    def save_best(self, policy_id: PolicyID, metric: str, metric_value: float) -> bool:
        if self.learner.save_best(policy_id, metric, metric_value):
            self.saved_model.emit(self.learner.policy_id)
            return True
        return False

    def save_milestone(self) -> None:
        self.learner.save_milestone()

    def load(self, policy_to_load: PolicyID) -> None:
        self.learner.set_policy_to_load(policy_to_load)

    def on_update_cfg(self, new_cfg: Dict) -> None:
        self.learner.set_new_cfg(new_cfg)

    def start_batcher_thread(self):
        self.batcher.event_loop.process = self.event_loop.process
        self.batcher_thread = Thread(target=self.batcher.event_loop.exec)
        self.batcher_thread.start()

    def join_batcher_thread(self):
        self.batcher_thread.join()

    def init(self):
        if not self.cfg.serial_mode:
            self.start_batcher_thread()

        init_model_data = self.learner.init()
        # signal other components that the model is ready
        self.model_initialized.emit(init_model_data)

        # runner should know the number of env steps in case we resume from a checkpoint
        self.report_msg.emit({LEARNER_ENV_STEPS: self.learner.env_steps, POLICY_ID_KEY: self.learner.policy_id})

        self.initialized.emit()
        log.debug(f"{self.object_id} finished initialization!")

    def on_new_training_batch(self, batch_idx: int):
        stats = self.learner.train(self.batcher.training_batches[batch_idx])

        self.training_iteration_since_resume += 1
        self.training_batch_released.emit(batch_idx, self.training_iteration_since_resume)
        self.finished_training_iteration.emit(self.training_iteration_since_resume)
        if stats is not None:
            self.report_msg.emit(stats)

    # noinspection PyMethodMayBeStatic
    def _cleanup_cache(self):
        torch.cuda.empty_cache()

    def on_stop(self, *args):
        self.learner.save()
        if not self.cfg.serial_mode:
            self.join_batcher_thread()

        self.stop.emit(self.object_id, {self.object_id: self.learner.timing})

        super().on_stop(*args)
        del self.learner.actor_critic
        del self.learner
