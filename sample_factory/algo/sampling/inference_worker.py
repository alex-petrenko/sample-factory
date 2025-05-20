from __future__ import annotations

import copy
import os
import time
from collections import deque
from queue import Empty
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from signal_slot.signal_slot import TightLoop, Timer, signal

from sample_factory.algo.utils.context import SampleFactoryContext, set_global_context
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.heartbeat import HeartbeatStoppableEventLoopObject
from sample_factory.algo.utils.misc import (
    POLICY_ID_KEY,
    SAMPLES_COLLECTED,
    STATS_KEY,
    TIMING_STATS,
    advance_rollouts_signal,
    memory_stats,
)
from sample_factory.algo.utils.model_sharing import ParameterServer, make_parameter_client
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.tensor_dict import TensorDict, to_numpy
from sample_factory.algo.utils.tensor_utils import cat_tensors, dict_of_lists_cat, ensure_torch_tensor
from sample_factory.algo.utils.torch_utils import inference_context, init_torch_runtime, synchronize
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.dicts import dict_of_lists_append_idx
from sample_factory.utils.gpu_utils import cuda_envvars_for_policy
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import Device, InitModelData, MpQueue, PolicyID
from sample_factory.utils.utils import debug_log_every_n, init_file_logger, log

AdvanceRolloutSignals = Dict[int, List[Tuple[int, PolicyID]]]
PrepareOutputsFunc = Callable[[int, TensorDict, List], AdvanceRolloutSignals]


def init_inference_process(sf_context: SampleFactoryContext, worker: InferenceWorker):
    set_global_context(sf_context)
    log.info(f"{worker.object_id}\tpid {os.getpid()}\tparent {os.getppid()}")

    # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
    import signal as os_signal

    os_signal.signal(os_signal.SIGINT, os_signal.SIG_IGN)

    cfg = worker.cfg
    init_file_logger(cfg)

    try:
        if cfg.num_workers > 1:
            psutil.Process().nice(min(cfg.default_niceness + 2, 20))
    except psutil.AccessDenied:
        log.error("Low niceness requires sudo!")

    if cfg.device == "gpu":
        cuda_envvars_for_policy(worker.policy_id, "inference")
    init_torch_runtime(cfg)


class InferenceWorker(HeartbeatStoppableEventLoopObject, Configurable):
    def __init__(
        self,
        event_loop,
        policy_id: PolicyID,
        worker_idx: int,
        buffer_mgr,
        param_server: ParameterServer,
        inference_queue: MpQueue,
        cfg,
        env_info: EnvInfo,
    ):
        Configurable.__init__(self, cfg)
        unique_name = f"{InferenceWorker.__name__}_p{policy_id}-w{worker_idx}"
        HeartbeatStoppableEventLoopObject.__init__(self, event_loop, unique_name, cfg.heartbeat_interval)

        self.timing = Timing(name=f"{self.object_id} profile")

        self.policy_id: PolicyID = policy_id
        self.worker_idx: int = worker_idx

        self.buffer_mgr = buffer_mgr

        # shallow copy
        self.traj_tensors: Dict[Device, TensorDict] = copy.copy(buffer_mgr.traj_tensors_torch)
        self.policy_output_tensors: Dict[Device, TensorDict] = copy.copy(buffer_mgr.policy_output_tensors_torch)

        self.device: torch.device = policy_device(cfg, policy_id)
        self.param_client = make_parameter_client(cfg.serial_mode, param_server, cfg, env_info, self.timing)
        self.inference_queue = inference_queue

        self.request_count = deque(maxlen=50)

        # very conservative limit on the minimum number of requests to wait for
        # this will almost guarantee that the system will continue collecting experience
        # at max rate even when 2/3 of workers are stuck for some reason (e.g. doing a long env reset)
        # Although if your workflow involves very lengthy operations that often freeze workers, it can be beneficial
        # to set min_num_requests to 1 (at a cost of potential inefficiency, i.e. policy worker will use very small
        # batches)
        min_num_requests = self.cfg.num_workers // (self.cfg.num_policies * self.cfg.policy_workers_per_policy)
        min_num_requests //= 3
        self.min_num_requests = max(1, min_num_requests)
        log.info(f"{self.object_id}: min num requests: %d", self.min_num_requests)

        self.requests = []
        self.total_num_samples = self.last_report_samples = 0

        self._get_inference_requests_func = (
            self._get_inference_requests_serial if cfg.serial_mode else self._get_inference_requests_async
        )

        self.inference_loop: Optional[Timer] = None  # zero delay timer
        self.report_timer: Optional[Timer] = None
        self.cache_cleanup_timer: Optional[Timer] = None

        # flag used by the runner to determine when the worker is ready
        self.is_ready = False

        # behavior configuration depending on whether we're in batched or non-batched sampling regime
        if cfg.batched_sampling:
            self._batch_func = self._batch_slices
            prepare_policy_outputs = self._prepare_policy_outputs_batched
        else:
            self._batch_func = self._batch_individual_steps
            prepare_policy_outputs = self._prepare_policy_outputs_non_batched

        self._prepare_policy_outputs_func: PrepareOutputsFunc = prepare_policy_outputs

        self.is_initialized = False

    @signal
    def initialized(self):
        ...

    @signal
    def report_msg(self):
        ...

    def init(self, init_model_data: Optional[InitModelData]):
        if self.is_initialized:
            return

        if "cpu" in self.traj_tensors:
            self.traj_tensors["cpu"] = to_numpy(self.traj_tensors["cpu"])
            self.policy_output_tensors["cpu"] = to_numpy(self.policy_output_tensors["cpu"])

        state_dict = None
        policy_version = 0
        if init_model_data is not None:
            policy_id, state_dict, self.device, policy_version = init_model_data
            if policy_id != self.policy_id:
                return

        self.param_client.on_weights_initialized(state_dict, self.device, policy_version)

        # we can create and connect Timers and EventLoopObjects here because they all interact within one loop
        self.inference_loop = TightLoop(self.event_loop)
        self.inference_loop.iteration.connect(self._run)

        self.report_timer = Timer(self.event_loop, 3.0)
        self.report_timer.timeout.connect(self._report_stats)

        self.cache_cleanup_timer = Timer(self.event_loop, 0.5)
        if not self.cfg.benchmark:
            self.cache_cleanup_timer.timeout.connect(self._cache_cleanup)

        # singal to main process (runner) that we're ready
        self.initialized.emit(self.policy_id, self.worker_idx)

        self.is_initialized = True

    def should_stop_experience_collection(self):
        debug_log_every_n(50, f"{self.object_id}: stopping experience collection")
        self.inference_loop.stop()

    def should_resume_experience_collection(self):
        debug_log_every_n(50, f"{self.object_id}: resuming experience collection")
        self.inference_loop.start()

    def _batch_slices(self, timing):
        with timing.add_time("deserialize"):
            obs = dict()
            rnn_states = []
            for actor_idx, split_idx, traj_idx, device in self.requests:
                # TODO: what should we do with data sampled on different devices
                traj_tensors = self.traj_tensors[device]
                dict_of_lists_append_idx(obs, traj_tensors["obs"], traj_idx)
                rnn_states.append(traj_tensors["rnn_states"][traj_idx])

        with timing.add_time("stack"):
            if len(rnn_states) == 1:
                for obs_key, tensor_list in obs.items():
                    obs[obs_key] = tensor_list[0]
                rnn_states = rnn_states[0]
            else:
                # cat() will fail if samples are on different devices
                # should we handle a situation where experience comes from multiple devices?
                # i.e. we use multiple GPUs for sampling but inference/learning is on a single GPU
                dict_of_lists_cat(obs)
                rnn_states = cat_tensors(rnn_states)

        return obs, rnn_states

    def _batch_individual_steps(self, timing):
        with timing.add_time("deserialize"):
            indices = []
            for request in self.requests:
                # TODO: what should we do with data sampled on different devices
                actor_idx, split_idx, request_data, device = request
                for env_idx, agent_idx, traj_buffer_idx, rollout_step in request_data:
                    index = [traj_buffer_idx, rollout_step]
                    indices.append(index)

            indices = tuple(np.array(indices).T)
            traj_tensors = self.traj_tensors[device]  # TODO: multiple sampling devices?
            observations = traj_tensors["obs"][indices]
            rnn_states = traj_tensors["rnn_states"][indices]

        with timing.add_time("stack"):
            for key, x in observations.items():
                observations[key] = ensure_torch_tensor(x)
            rnn_states = ensure_torch_tensor(rnn_states)

        return observations, rnn_states

    @staticmethod
    def _unsqueeze_0dim_tensors(d: TensorDict):
        for policy_output in d.values():
            if not policy_output.shape:
                policy_output.unsqueeze_(-1)

    def _prepare_policy_outputs_batched(
        self, num_samples: int, policy_outputs: TensorDict, requests: List
    ) -> AdvanceRolloutSignals:
        # gotta unsqueeze some 0-dim tensors
        if num_samples <= 1:
            self._unsqueeze_0dim_tensors(policy_outputs)

        # actions are handled differently in the batched version so we have to convert them to
        # [batch_size, num_actions]
        if policy_outputs["actions"].ndim < 2:
            policy_outputs["actions"] = policy_outputs["actions"].unsqueeze(-1)

        # assuming all workers provide the same number of samples
        samples_per_actor = num_samples // len(requests)
        ofs = 0
        devices_to_sync = set()
        for actor_idx, split_idx, _, device in requests:
            self.policy_output_tensors[device][actor_idx, split_idx] = policy_outputs[ofs : ofs + samples_per_actor]
            ofs += samples_per_actor
            devices_to_sync.add(device)

        signals_to_send: AdvanceRolloutSignals = dict()
        for actor_idx, split_idx, _, _ in requests:
            payload = (split_idx, self.policy_id)
            if actor_idx in signals_to_send:
                signals_to_send[actor_idx].append(payload)
            else:
                signals_to_send[actor_idx] = [payload]

        # to make sure we committed all writes to shared device memory, we need to sync all devices
        # typically this will be a single CUDA device
        for device in devices_to_sync:
            synchronize(self.cfg, device)

        return signals_to_send

    def _prepare_policy_outputs_non_batched(
        self, _num_samples: int, policy_outputs: TensorDict, requests: List
    ) -> AdvanceRolloutSignals:
        # Respect sampling device instead of just dumping everything on cpu?
        # Although it is hard to imagine a scenario where we have a non-batched env with observations on gpu
        device = "cpu"

        with self.timing.add_time("to_cpu"):
            for key, output_value in policy_outputs.items():
                policy_outputs[key] = output_value.to(device)

        # concat all tensors into a single tensor for performance
        output_tensors = []
        for name in self.buffer_mgr.output_names:
            output_value = policy_outputs[name].float()
            while output_value.dim() <= 1:
                output_value.unsqueeze_(-1)
            output_tensors.append(output_value)

        output_tensors = torch.cat(output_tensors, dim=1)

        signals_to_send: AdvanceRolloutSignals = dict()
        output_indices = []
        for request in requests:
            actor_idx, split_idx, request_data, _ = request
            for env_idx, agent_idx, traj_buffer_idx, rollout_step in request_data:
                output_indices.append([actor_idx, split_idx, env_idx, agent_idx])

            payload = (split_idx, self.policy_id)
            if actor_idx in signals_to_send:
                signals_to_send[actor_idx].append(payload)
            else:
                signals_to_send[actor_idx] = [payload]

        output_indices = tuple(np.array(output_indices).T)
        self.policy_output_tensors[device][output_indices] = output_tensors.numpy()

        # this should be a no-op unless we have a non-batched env with observations on gpu
        synchronize(self.cfg, device)

        return signals_to_send

    def _handle_policy_steps(self, timing):
        with inference_context(self.cfg.serial_mode):
            obs, rnn_states = self._batch_func(timing)
            num_samples = rnn_states.shape[0]
            self.total_num_samples += num_samples

            with timing.add_time("obs_to_device_normalize"):
                actor_critic = self.param_client.actor_critic
                if actor_critic.training:
                    actor_critic.eval()  # need to call this because we can be in serial mode

                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                rnn_states = ensure_torch_tensor(rnn_states).to(self.device).float()

            with timing.add_time("forward"):
                policy_outputs = actor_critic(normalized_obs, rnn_states)
                policy_outputs["policy_version"] = torch.empty([num_samples]).fill_(self.param_client.policy_version)

            with timing.add_time("prepare_outputs"):
                signals_to_send = self._prepare_policy_outputs_func(num_samples, policy_outputs, self.requests)

            with timing.add_time("send_messages"):
                for actor_idx, data in signals_to_send.items():
                    self.emit_many(advance_rollouts_signal(actor_idx), data)

            self.requests = []

    def _get_inference_requests_serial(self):
        try:
            self.requests.extend(self.inference_queue.get_many(block=False))
        except Empty:
            pass

    def _get_inference_requests_async(self):
        # Very conservative timer. Only wait a little bit, then continue with what we've got.
        wait_for_min_requests = 0.025

        waiting_started = time.time()
        while len(self.requests) < self.min_num_requests and time.time() - waiting_started < wait_for_min_requests:
            try:
                with self.timing.timeit("wait_policy"), self.timing.add_time("wait_policy_total"):
                    policy_requests = self.inference_queue.get_many(timeout=0.005)
                self.requests.extend(policy_requests)
            except Empty:
                pass

    def _run(self):
        self._get_inference_requests_func()
        if not self.requests:
            return

        with self.timing.add_time("update_model"):
            self.param_client.ensure_weights_updated()

        with self.timing.timeit("one_step"), self.timing.add_time("handle_policy_step"):
            self.request_count.append(len(self.requests))
            self._handle_policy_steps(self.timing)

    def _report_stats(self):
        if "one_step" not in self.timing:
            return

        timing_stats = dict(wait_policy=self.timing.get("wait_policy", 0), step_policy=self.timing.one_step)
        samples_since_last_report = self.total_num_samples - self.last_report_samples
        self.last_report_samples = self.total_num_samples

        stats = memory_stats("policy_worker", self.device)
        if len(self.request_count) > 0:
            stats["avg_request_count"] = np.mean(self.request_count)

        self.report_msg.emit(
            {
                TIMING_STATS: timing_stats,
                SAMPLES_COLLECTED: samples_since_last_report,
                POLICY_ID_KEY: self.policy_id,
                STATS_KEY: stats,
            }
        )

    def _cache_cleanup(self):
        if self.cfg.device == "gpu":
            torch.cuda.empty_cache()

        # initially we clean cache very frequently, later on do it every few minutes
        if self.total_num_samples > 1000:
            self.cache_cleanup_timer.set_interval(60.0)

    def on_stop(self, *args):
        if self.is_initialized:
            self.param_client.cleanup()
            del self.param_client

        self.stop.emit(self.object_id, {self.object_id: self.timing})
        super().on_stop(*args)
