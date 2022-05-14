import os
import time
from collections import deque
from queue import Empty
from typing import Dict, Any, List, Optional

import numpy as np
import psutil
import torch

from sample_factory.algo.utils.context import SampleFactoryContext, set_global_context
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import make_parameter_client, ParameterServer
from sample_factory.algo.utils.torch_utils import init_torch_runtime, inference_context
from sample_factory.algorithms.appo.appo_utils import cuda_envvars_for_policy, memory_stats
from sample_factory.algorithms.appo.shared_buffers import TensorDict
from sample_factory.cfg.configurable import Configurable
from sample_factory.signal_slot.signal_slot import EventLoopObject, Timer, TightLoop, signal
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import PolicyID, MpQueue
from sample_factory.utils.utils import log


def init_sampler_process(sf_context: SampleFactoryContext, cfg, policy_id):
    set_global_context(sf_context)
    log.info(f'INFERENCE worker {policy_id}\tpid {os.getpid()}\tparent {os.getppid()}')

    # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
    import signal as os_signal
    os_signal.signal(os_signal.SIGINT, os_signal.SIG_IGN)

    try:
        psutil.Process().nice(min(cfg.default_niceness + 2, 20))
    except psutil.AccessDenied:
        log.error('Low niceness requires sudo!')

    if cfg.device == 'gpu':
        cuda_envvars_for_policy(policy_id, 'inference')
    init_torch_runtime(cfg)


def dict_of_lists_append(d: Dict[Any, List], new_data, index):
    for key, x in new_data.items():
        if key in d:
            d[key].append(x[index])
        else:
            d[key] = [x[index]]


class InferenceWorker(EventLoopObject, Configurable):
    def __init__(
            self, event_loop, policy_id: PolicyID, worker_idx: int, buffer_mgr,
            param_server: ParameterServer, inference_queue: MpQueue, cfg, env_info: EnvInfo,
    ):
        Configurable.__init__(self, cfg)
        unique_name = f'{InferenceWorker.__name__}_p{policy_id}-w{worker_idx}'
        EventLoopObject.__init__(self, event_loop, unique_name)

        self.timing = Timing(name=f'{self.object_id} profile')

        self.policy_id: PolicyID = policy_id
        self.worker_idx: int = worker_idx

        self.buffer_mgr = buffer_mgr
        self.traj_tensors = buffer_mgr.traj_tensors
        self.obs_tensors, self.rnn_state_tensors = self.traj_tensors['obs'], self.traj_tensors['rnn_states']
        self.policy_output_tensors = buffer_mgr.policy_output_tensors

        self.device: Optional[torch.device] = None
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
        log.info(f'{self.object_id}: min num requests: %d', self.min_num_requests)

        self.requests = []
        self.total_num_samples = self.last_report_samples = 0

        self._get_inference_requests_func = self._get_inference_requests_serial if cfg.serial_mode else self._get_inference_requests_serial

        self.inference_loop: Optional[Timer] = None  # zero delay timer
        self.report_timer: Optional[Timer] = None
        self.cache_cleanup_timer: Optional[Timer] = None

        # flag used by the runner to determine when the worker is ready
        self.is_ready = False

    @signal
    def initialized(self): pass

    @signal
    def report_msg(self): pass

    @signal
    def stop(self): pass

    def init(self, initial_model_state):
        state_dict, self.device, policy_version = initial_model_state
        self.param_client.on_weights_initialized(state_dict, self.device, policy_version)

        # we can create and connect Timers and EventLoopObjects here because they all interact within one loop
        self.inference_loop = TightLoop(self.event_loop)
        self.inference_loop.iteration.connect(self._run)

        self.report_timer = Timer(self.event_loop, 3.0)
        self.report_timer.timeout.connect(self._report_stats)

        self.cache_cleanup_timer = Timer(self.event_loop, 0.1)
        if not self.cfg.benchmark:
            self.cache_cleanup_timer.timeout.connect(self._cache_cleanup)

        # singal to main process (runner) that we're ready
        self.initialized.emit(self.policy_id, self.worker_idx)

    def _handle_policy_steps(self, timing):
        # TODO: batch requests together. Two cases 1) numpy indices 2) torch tensor slices
        # (self.worker_idx, split_idx, requests)/
        # (self.curr_traj_slice, self.rollout_step)

        with inference_context(self.cfg.serial_mode):
            with timing.add_time('deserialize'):
                obs = dict()
                rnn_states = []
                for actor_idx, split_idx, traj_idx in self.requests:
                    dict_of_lists_append(obs, self.obs_tensors, traj_idx)
                    rnn_states.append(self.rnn_state_tensors[traj_idx])

            with timing.add_time('stack'):
                # TODO: do we need an extra case where it is just one big batch? So we don't need to call cat
                if len(rnn_states) == 1:
                    for obs_key, tensor_list in obs.items():
                        obs[obs_key] = tensor_list[0]
                    rnn_states = rnn_states[0]
                else:
                    for obs_key, tensor_list in obs.items():
                        obs[obs_key] = torch.cat(tensor_list)
                    rnn_states = torch.cat(rnn_states)

                num_samples = rnn_states.shape[0]
                self.total_num_samples += num_samples

            with timing.add_time('obs_to_device'):
                actor_critic = self.param_client.actor_critic
                with timing.add_time('model_eval'):
                    actor_critic.eval()  # need to call this because we can be in serial mode

                for key, x in obs.items():
                    device, dtype = actor_critic.device_and_type_for_input_tensor(key)
                    obs[key] = x.to(device).type(dtype)
                rnn_states = rnn_states.to(self.device).float()

            with self.timing.add_time('norm'):
                normalized_obs = actor_critic.normalizer(obs)

            with timing.add_time('forward'):
                policy_outputs = actor_critic(normalized_obs, rnn_states)

            with timing.add_time('save_outputs'):
                policy_outputs['policy_version'] = torch.empty([num_samples]).fill_(self.param_client.policy_version)

                # assuming all workers provide the same number of samples
                samples_per_request = num_samples // len(self.requests)
                ofs = 0
                env_idx = 0  # with batched sampling we only have one vector env per "split"
                for actor_idx, split_idx, _ in self.requests:
                    self.policy_output_tensors[actor_idx, split_idx, env_idx] = policy_outputs[ofs:ofs + samples_per_request]
                    ofs += samples_per_request

            with timing.add_time('send_messages'):
                for actor_idx, split_idx, _ in self.requests:
                    self.emit(f'advance{actor_idx}', (split_idx, self.policy_id))

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
                with self.timing.timeit('wait_policy'), self.timing.add_time('wait_policy_total'):
                    policy_requests = self.inference_queue.get_many(timeout=0.005)
                self.requests.extend(policy_requests)
            except Empty:
                pass

    def _run(self):
        # TODO: termination

        # TODO: implement this or a replacement mechanism (probably just emit a signal on the learner?)
        # while self.shared_buffers.stop_experience_collection[self.policy_id]:
        #     with self.resume_experience_collection_cv:
        #         self.resume_experience_collection_cv.wait(timeout=0.05)

        self._get_inference_requests_func()
        if not self.requests:
            return

        with self.timing.add_time('update_model'):
            self.param_client.ensure_weights_updated()

        with self.timing.timeit('one_step'), self.timing.add_time('handle_policy_step'):
            self.request_count.append(len(self.requests))  # TODO: count requests properly when we use slices
            self._handle_policy_steps(self.timing)

    def _report_stats(self):
        if 'one_step' not in self.timing:
            return

        timing_stats = dict(wait_policy=self.timing.get('wait_policy', 0), step_policy=self.timing.one_step)
        samples_since_last_report = self.total_num_samples - self.last_report_samples

        stats = memory_stats('policy_worker', self.device)
        if len(self.request_count) > 0:
            stats['avg_request_count'] = np.mean(self.request_count)

        self.report_msg.emit()

        self.report_msg.emit(dict(
            timing=timing_stats, samples=samples_since_last_report, policy_id=self.policy_id, stats=stats,
        ))

    def _cache_cleanup(self):
        if self.cfg.device == 'gpu':
            torch.cuda.empty_cache()

        # initially we clean cache very frequently, later on do it every few minutes
        if self.total_num_samples > 1000:
            self.cache_cleanup_timer.set_interval(300.0)

    def on_stop(self, emitter_id):
        log.debug(f'Stopping {self.object_id}...')

        self.stop.emit(self.object_id)

        if self.event_loop.owner is self:
            self.event_loop.stop()

        self.detach()  # remove from the current event loop
        log.info(self.timing)
