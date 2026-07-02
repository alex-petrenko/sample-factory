from __future__ import annotations

import math
import time
from collections import OrderedDict
from threading import Thread
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from signal_slot.signal_slot import EventLoop, EventLoopObject, EventLoopStatus, signal
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.runners.runner import MsgHandler, PolicyMsgHandler
from sample_factory.algo.sampling.sampler import AbstractSampler, ParallelSampler, SerialSampler
from sample_factory.algo.sampling.stats import samples_stats_handler, stats_msg_handler, timing_msg_handler
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import EPISODIC, SAMPLES_COLLECTED, STATS_KEY, TIMING_STATS, ExperimentStatus
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import samples_per_trajectory
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.arguments import cfg_dict
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.gpu_utils import set_global_cuda_envvars
from sample_factory.utils.typing import Config, InitModelData, PolicyID, StatusCode
from sample_factory.utils.utils import log
from sample_factory.utils.state_proxy import StateProxy


class SamplingLoop(StateProxy, EventLoopObject, Configurable):
    _STATE_ATTRS = frozenset(('avg_stats', 'buffer_mgr', 'env_info', 'event_loop', 'iteration', 'max_episode_number', 'msg_handlers', 'new_trajectory_callback', 'param_servers', 'policy_avg_stats', 'policy_msg_handlers', 'print_episode_info', 'ready', 'samples_collected', 'stats', 'status', 'stopped'))

    def __init__(self, cfg: Config, env_info: EnvInfo, print_episode_info: bool = True):
        self._init_state_proxy()
        Configurable.__init__(self, cfg_dict(cfg))

        unique_name = SamplingLoop.__name__
        self._state.event_loop: EventLoop = EventLoop(unique_loop_name=f"{unique_name}_EvtLoop", serial_mode=cfg.serial_mode)
        self._state.event_loop.owner = self
        EventLoopObject.__init__(self, self._state.event_loop, object_id=unique_name)
        # self._state.event_loop.verbose = True

        # calculate how many episodes for each environment should be taken into account
        # we only want to use first N episodes (we don't want to bias ourselves with short episodes)
        total_envs = self.cfg.num_workers * self.cfg.num_envs_per_worker

        sample_env_episodes = self.cfg.get("sample_env_episodes", math.inf)
        self._state.max_episode_number = sample_env_episodes / total_envs

        self._state.env_info = env_info
        self._state.iteration: int = 0

        self._state.buffer_mgr: Optional[BufferMgr] = None
        self._state.param_servers: Optional[Dict[PolicyID, ParameterServer]] = None

        self._state.new_trajectory_callback: Optional[Callable] = None
        self._state.status: Optional[StatusCode] = None

        self._state.ready: bool = False
        self._state.stopped: bool = False

        # samples_collected counts the total number of observations processed by the algorithm
        self._state.samples_collected = [0 for _ in range(self.cfg.num_policies)]

        self._state.stats = dict()  # regular (non-averaged) stats
        self._state.avg_stats = dict()

        self._state.policy_avg_stats: Dict[str, List[List]] = dict()

        # global msg handlers for messages from algo components
        self._state.msg_handlers: Dict[str, List[MsgHandler]] = {
            TIMING_STATS: [timing_msg_handler],
            STATS_KEY: [stats_msg_handler],
        }

        # handlers for policy-specific messages
        self._state.policy_msg_handlers: Dict[str, List[PolicyMsgHandler]] = {
            EPISODIC: [self._episodic_stats_handler],
            SAMPLES_COLLECTED: [samples_stats_handler],
        }

        self._state.print_episode_info = print_episode_info

    @signal
    def model_initialized(self): ...

    @signal
    def trajectory_buffers_available(self): ...

    @signal
    def stop(self): ...

    def init(
        self, buffer_mgr: Optional[BufferMgr] = None, param_servers: Optional[Dict[PolicyID, ParameterServer]] = None
    ):
        set_global_cuda_envvars(self.cfg)

        self._state.buffer_mgr = buffer_mgr
        if self._state.buffer_mgr is None:
            self._state.buffer_mgr = BufferMgr(self.cfg, self._state.env_info)

        self._state.param_servers = param_servers
        if self._state.param_servers is None:
            self._state.param_servers = dict()
            for policy_id in range(self.cfg.num_policies):
                self._state.param_servers[policy_id] = ParameterServer(
                    policy_id, self._state.buffer_mgr.policy_versions, self.cfg.serial_mode
                )

        sampler_cls = SerialSampler if self.cfg.serial_mode else ParallelSampler
        sampler: AbstractSampler = sampler_cls(
            self._state.event_loop, self._state.buffer_mgr, self._state.param_servers, self.cfg, self._state.env_info
        )
        self._state.event_loop.start.connect(sampler.init)
        sampler.started.connect(self.on_sampler_started)
        sampler.initialized.connect(self.on_sampler_initialized)

        for policy_id in range(self.cfg.num_policies):
            sampler.connect_model_initialized(policy_id, self.model_initialized)
            sampler.connect_on_new_trajectories(policy_id, self.on_new_trajectories)
        sampler.connect_trajectory_buffers_available(self.trajectory_buffers_available)
        sampler.connect_report_msg(self._process_msg)

        for stoppable in sampler.stoppable_components():
            self.stop.connect(stoppable.on_stop)

    def _process_msg(self, msgs):
        if isinstance(msgs, (dict, OrderedDict)):
            msgs = (msgs,)

        if not (isinstance(msgs, (List, Tuple)) and isinstance(msgs[0], (dict, OrderedDict))):
            log.error("While parsing a message: expected a dictionary or list/tuple of dictionaries, found %r", msgs)
            return

        for msg in msgs:
            # some messages are policy-specific
            policy_id = msg.get("policy_id", None)

            for key in msg:
                for handler in self._state.msg_handlers.get(key, ()):
                    handler(self, msg)
                if policy_id is not None:
                    for handler in self._state.policy_msg_handlers.get(key, ()):
                        handler(self, msg, policy_id)

    @staticmethod
    def _episodic_stats_handler(stats_observer: SamplingLoop, msg: Dict, policy_id: PolicyID) -> None:
        # heavily based on the `_episodic_stats_handler` from `Runner`
        s = msg[EPISODIC]

        # skip invalid stats, potentially be not setting episode_number one could always add stats
        episode_number = s["episode_extra_stats"].get("episode_number", 0)
        if episode_number < stats_observer.max_episode_number:
            if stats_observer.print_episode_info:
                log.debug(
                    f"Episode ended after {s['len']:.1f} steps. Return: {s['reward']:.1f}. True objective {s['true_objective']:.1f}"
                )

            for _, key, value in iterate_recursively(s):
                if key not in stats_observer.policy_avg_stats:
                    stats_observer.policy_avg_stats[key] = [[] for _ in range(stats_observer.cfg.num_policies)]

                if isinstance(value, np.ndarray) and value.ndim > 0:
                    stats_observer.policy_avg_stats[key][policy_id].extend(value)
                else:
                    stats_observer.policy_avg_stats[key][policy_id].append(value)

    def wait_until_ready(self):
        while not self._state.ready:
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
        self._state.new_trajectory_callback = cb

    def on_sampler_started(self):
        self._state.ready = True

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
            traj = self._state.buffer_mgr.traj_tensors_torch[device][trajectory_slice]
            self._state.new_trajectory_callback(traj, [traj_buffer_idx], device)

    def yield_trajectory_buffers(self, available_buffers: Iterable[int | slice], device: str):
        # make this trajectory buffer available again
        self._state.buffer_mgr.traj_buffer_queues[device].put_many(available_buffers)
        self._state.iteration += 1
        for policy_id in range(self.cfg.num_policies):
            self.trajectory_buffers_available.emit(policy_id, self._state.iteration)

    def run(self) -> StatusCode:
        log.debug("Before event loop...")

        # noinspection PyBroadException
        try:
            evt_loop_status = self._state.event_loop.exec()
            self._state.status = (
                ExperimentStatus.INTERRUPTED
                if evt_loop_status == EventLoopStatus.INTERRUPTED
                else ExperimentStatus.SUCCESS
            )
            self.stop.emit()
        except Exception:
            log.exception(f"Uncaught exception in {self.object_id} evt loop")
            self._state.status = ExperimentStatus.FAILURE

        log.debug(f"{SamplingLoop.__name__} finished with {self._state.status=}")
        return self._state.status

    def stop_sampling(self):
        self.stop.emit()
        self._state.event_loop.stop()
        self._state.stopped = True


class EvalSamplingAPI(StateProxy):
    _STATE_ATTRS = frozenset(('buffer_mgr', 'cfg', 'env_info', 'init_model_data', 'learners', 'param_servers', 'policy_versions_tensor', 'sampling_loop', 'sampling_thread', 'total_samples'))

    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
    ):
        self._init_state_proxy()
        self._state.cfg = cfg
        self._state.env_info = env_info

        self._state.buffer_mgr = None
        self._state.policy_versions_tensor = None
        self._state.param_servers: Optional[dict[PolicyID, ParameterServer]] = None
        self._state.init_model_data: Optional[dict[PolicyID, InitModelData]] = None
        self._state.learners: Optional[dict[PolicyID, Learner]] = None

        self._state.sampling_loop: Optional[SamplingLoop] = None

        self._state.sampling_thread: Optional[Thread] = None

        self._state.total_samples = 0

    def init(self):
        set_global_cuda_envvars(self._state.cfg)

        self._state.buffer_mgr = BufferMgr(self._state.cfg, self._state.env_info)
        self._state.policy_versions_tensor: Tensor = self._state.buffer_mgr.policy_versions

        self._state.param_servers = {}
        self._state.init_model_data = {}
        self._state.learners = {}
        for policy_id in range(self._state.cfg.num_policies):
            self._state.param_servers[policy_id] = ParameterServer(
                policy_id, self._state.policy_versions_tensor, self._state.cfg.serial_mode
            )
            self._state.learners[policy_id] = Learner(
                self._state.cfg, self._state.env_info, self._state.policy_versions_tensor, policy_id, self._state.param_servers[policy_id]
            )
            # TODO: separate model loading from the learners
            self._state.init_model_data[policy_id] = self._state.learners[policy_id].init()

        self._state.sampling_loop: SamplingLoop = SamplingLoop(self._state.cfg, self._state.env_info)
        # don't pass self._state.param_servers here, learners are normally initialized later
        # TODO: fix above issue
        self._state.sampling_loop.init(self._state.buffer_mgr)
        self._state.sampling_loop.set_new_trajectory_callback(self._on_new_trajectories)
        self._state.sampling_thread = Thread(target=self._state.sampling_loop.run)
        self._state.sampling_thread.start()

        self._state.sampling_loop.wait_until_ready()

    @property
    def eval_stats(self):
        # it's possible that we would like to return additional stats, like fps or sth
        # those could be added here
        return self._state.sampling_loop.policy_avg_stats

    @property
    def eval_episodes(self):
        return self.eval_stats.get("episode_number", [[] for _ in range(self._state.cfg.num_policies)])

    @property
    def eval_env_steps(self):
        # return number of env steps for each policy
        episode_lens = self.eval_stats.get("len", [[] for _ in range(self._state.cfg.num_policies)])
        return [sum(episode_lens[policy_id]) for policy_id in range(self._state.cfg.num_policies)]

    def start(self, init_model_data: Optional[Dict[PolicyID, InitModelData]] = None):
        if init_model_data is None:
            init_model_data = self._state.init_model_data
        self._state.sampling_loop.start(init_model_data)

    def _on_new_trajectories(self, traj: TensorDict, traj_buffer_indices: Iterable[int | slice], device: str):
        self._state.total_samples += samples_per_trajectory(traj)

        # just release buffers after every trajectory
        # we could alternatively have more sophisticated logic here, see i.e. batcher.py or sync_sampling_api.py
        self._state.sampling_loop.yield_trajectory_buffers(traj_buffer_indices, device)

    def stop(self) -> StatusCode:
        self._state.sampling_loop.stop_sampling()
        self._state.sampling_thread.join()
        return self._state.sampling_loop.status
