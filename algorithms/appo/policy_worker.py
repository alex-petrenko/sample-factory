import math
import multiprocessing
import signal
import time
from collections import deque
from queue import Empty

import faster_fifo
import numpy as np
import psutil
import torch
from gym import spaces
from torch.multiprocessing import Process as TorchProcess

from algorithms.appo.appo_utils import TaskType, memory_stats, cuda_envvars_for_policy
from algorithms.appo.model import create_actor_critic
from algorithms.appo.model_utils import get_hidden_size
from algorithms.appo.shared_buffers import to_torch_dtype, TensorDict
from utils.timing import Timing
from utils.utils import AttrDict, log, join_or_kill


def dict_of_lists_append(dict_of_lists, new_data, index):
    for key, x in new_data.items():
        if key in dict_of_lists:
            dict_of_lists[key].append(x[index])
        else:
            dict_of_lists[key] = [x[index]]


def _init_process(process_name, cfg, policy_id):
    # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    psutil.Process().nice(min(cfg.default_niceness + 2, 20))

    cuda_envvars_for_policy(policy_id, process_name)
    torch.multiprocessing.set_sharing_strategy('file_system')

    torch.set_num_threads(1)

    if cfg.device == 'gpu':
        # we should already see only one CUDA device, because of env vars
        assert torch.cuda.device_count() == 1
        device = torch.device('cuda', index=0)
    else:
        device = torch.device('cpu')

    return device


class _InferenceWorkerBase:
    def __init__(self, worker_idx, policy_id, cfg, obs_space, action_space, shared_buffers):
        self.worker_idx = worker_idx
        self.policy_id = policy_id
        self.cfg = cfg

        self.obs_space = obs_space
        self.action_space = action_space

        self.shared_buffers = shared_buffers
        self.tensors_individual_transitions = self.shared_buffers.tensors_individual_transitions
        self.policy_versions = shared_buffers.policy_versions
        self.stop_experience_collection = shared_buffers.stop_experience_collection

        self.device = None

        self.process = TorchProcess(target=self._run, daemon=True)

    def _run(self):
        raise NotImplementedError('Override me')


class _InferenceWorkerProcess(_InferenceWorkerBase):
    def __init__(
        self, worker_idx, policy_id, cfg, obs_space, action_space, shared_buffers, actor_queues, policy_lock,
        inference_torch_queue, inference_fast_queue, free_buffer_queue,
    ):
        super().__init__(worker_idx, policy_id, cfg, obs_space, action_space, shared_buffers)

        self.actor_queues = actor_queues
        self.policy_lock = policy_lock

        self.inference_torch_queue = inference_torch_queue
        self.inference_fast_queue = inference_fast_queue

        self.device_obs = self.device_rnn_states = None
        self.free_buffer_queue = free_buffer_queue

        self.actor_critic = None
        self.shared_model_weights = None

        self.latest_policy_version = -1
        self.num_policy_updates = 0

    def _run(self):
        timing = Timing()
        batch_sizes = deque(maxlen=50)
        total_num_samples = 0
        last_cache_cleanup = time.time()

        self.device = _init_process(f'inference_main_{self.policy_id}_{self.worker_idx}', self.cfg, self.policy_id)

        log.info('Initializing model on the policy worker %d-%d...', self.policy_id, self.worker_idx)

        self.actor_critic = create_actor_critic(self.cfg, self.obs_space, self.action_space, timing)
        self.actor_critic.model_to_device(self.device)
        for p in self.actor_critic.parameters():
            p.requires_grad = False  # we don't train anything here

        log.info('Initialized model on the policy worker %d-%d!', self.policy_id, self.worker_idx)

        while True:
            msg_type, data = self.inference_torch_queue.get()
            if msg_type == TaskType.INIT:
                self.device_obs, self.device_rnn_states = data
            elif msg_type == TaskType.INIT_MODEL:
                self._init_model(data)

            if self.latest_policy_version >= 0 and self.device_obs is not None:
                log.debug('Inference process %d-%d initialized!', self.policy_id, self.worker_idx)
                break

        while True:
            with timing.add_time('update_weights'):
                self._update_weights(timing)

            try:
                msg = self.inference_fast_queue.get(timeout=0.1)
                if msg is None:
                    break
            except Empty:
                continue

            buffer_idx, batch_size, requests = msg
            batch_sizes.append(batch_size)

            total_num_samples += batch_size

            with timing.add_time('inference'):
                self._inference(buffer_idx, batch_size, requests, timing)

            if time.time() - last_cache_cleanup > 300.0 or (not self.cfg.benchmark and total_num_samples < 1000):
                if self.cfg.device == 'gpu':
                    torch.cuda.empty_cache()
                last_cache_cleanup = time.time()

        log.info('Policy worker inference loop: avg. batch size %.1f, timing: %s', np.mean(batch_sizes), timing)

    def _inference(self, buffer_idx, batch_size, requests, timing):
        with timing.add_time('forward'):
            i = buffer_idx
            policy_outputs = self.actor_critic(
                self.device_obs[i].slice(0, batch_size), self.device_rnn_states[i][0:batch_size],
            )

        with timing.add_time('format_output_tensors'):
            self.free_buffer_queue.put(buffer_idx)

            # for key, output_value in policy_outputs.items():
            #     policy_outputs[key] = output_value.cpu()

            policy_outputs.policy_version = torch.empty([batch_size], device=self.device).fill_(self.latest_policy_version)

            with timing.add_time('concat'):
                # concat all tensors into a single tensor for performance
                output_tensors = []
                for policy_output in self.shared_buffers.policy_outputs:
                    tensor_name = policy_output.name
                    output_value = policy_outputs[tensor_name].float()
                    if len(output_value.shape) == 1:
                        output_value.unsqueeze_(dim=1)
                    output_tensors.append(output_value)

                output_tensors = torch.cat(output_tensors, dim=1)

            with timing.add_time('to_cpu'):
                output_tensors = output_tensors.cpu()

        with timing.add_time('postprocess'):
            self._enqueue_policy_outputs(requests, output_tensors)

    def _enqueue_policy_outputs(self, requests, output_tensors):
        output_idx = 0
        outputs_ready = set()

        policy_outputs = self.shared_buffers.policy_output_tensors

        for request in requests:
            actor_idx, split_idx, request_data = request
            worker_outputs = policy_outputs[actor_idx, split_idx]
            for env_idx, agent_idx, traj_buffer_idx, rollout_step in request_data:
                worker_outputs[env_idx, agent_idx].copy_(output_tensors[output_idx])
                output_idx += 1
            outputs_ready.add((actor_idx, split_idx))

        for actor_idx, split_idx in outputs_ready:
            advance_rollout_request = dict(split_idx=split_idx, policy_id=self.policy_id)
            self.actor_queues[actor_idx].put((TaskType.ROLLOUT_STEP, advance_rollout_request))

    def _update_weights(self, timing):
        learner_policy_version = self.policy_versions[self.policy_id].item()
        if self.latest_policy_version < learner_policy_version and self.shared_model_weights is not None:
            with timing.timeit('weight_update'):
                with self.policy_lock:
                    self.actor_critic.load_state_dict(self.shared_model_weights)

            self.latest_policy_version = learner_policy_version

            if self.num_policy_updates % 10 == 0:
                log.info(
                    'Updated weights on worker %d-%d, policy_version %d (%.5f)',
                    self.policy_id, self.worker_idx, self.latest_policy_version, timing.weight_update,
                )

            self.num_policy_updates += 1

    def _init_model(self, init_model_data):
        policy_version, state_dict = init_model_data
        self.actor_critic.load_state_dict(state_dict)
        self.shared_model_weights = state_dict
        self.latest_policy_version = policy_version


class PolicyWorker(_InferenceWorkerBase):
    def __init__(
        self, worker_idx, policy_id, cfg, obs_space, action_space, shared_buffers,
        policy_queue, actor_queues, report_queue, inference_queue,
        policy_lock, resume_experience_collection_cv,
    ):
        super().__init__(worker_idx, policy_id, cfg, obs_space, action_space, shared_buffers)

        log.info('Initializing policy worker %d for policy %d', worker_idx, policy_id)

        self.resume_experience_collection_cv = resume_experience_collection_cv

        self.policy_queue = policy_queue
        self.actor_queues = actor_queues
        self.report_queue = report_queue

        # queue other components use to talk to this particular worker
        self.task_queue = torch.multiprocessing.JoinableQueue()

        self.initialized = False
        self.terminate = False
        self.initialized_event = multiprocessing.Event()
        self.initialized_event.clear()

        self.total_num_samples = 0

        # self.max_inference_batch = self.cfg.num_workers * self.cfg.num_envs_per_worker // self.cfg.worker_num_splits
        self.max_inference_batch = self.cfg.num_workers * self.cfg.num_envs_per_worker  # TODO
        self.num_device_buffers = 2  # two should be enough for double-buffering

        self.free_buffer_queue = faster_fifo.Queue()
        for i in range(self.num_device_buffers):
            self.free_buffer_queue.put(i)

        self.device_obs = [TensorDict() for _ in range(self.num_device_buffers)]
        self.device_rnn_states = []

        self.inference_queue = inference_queue
        self.inference_fast_queue = faster_fifo.Queue()

        self.inference_worker = _InferenceWorkerProcess(
            worker_idx, policy_id, cfg, obs_space, action_space, shared_buffers, actor_queues, policy_lock,
            inference_queue, self.inference_fast_queue, self.free_buffer_queue,
        )

    def start_process(self):
        self.process.start()
        self.inference_worker.process.start()

    def _init(self):
        log.info('Policy worker %d-%d initialized', self.policy_id, self.worker_idx)
        self.initialized = True
        self.initialized_event.set()

    def _handle_policy_steps(self, requests, timing):
        with torch.no_grad():
            with timing.add_time('deserialize'):
                observations = AttrDict()
                rnn_states = []

                traj_tensors = self.shared_buffers.tensors_individual_transitions
                num_obs = 0

                for request in requests:
                    actor_idx, split_idx, request_data = request

                    for env_idx, agent_idx, traj_buffer_idx, rollout_step in request_data:
                        index = actor_idx, split_idx, env_idx, agent_idx, traj_buffer_idx, rollout_step
                        dict_of_lists_append(observations, traj_tensors['obs'], index)
                        rnn_states.append(traj_tensors['rnn_states'][index])
                        self.total_num_samples += 1

                        # with timing.add_time('copy_to_gpu'):
                        #     if num_obs < self.max_inference_batch:
                        #         for key, x in traj_tensors['obs'].items():
                        #             self.device_buffers[key][num_obs] = x[index]

                        num_obs += 1

            with timing.add_time('wait_device_buffer'):
                free_buffer = math.nan

                while not self.terminate:
                    try:
                        free_buffer = self.free_buffer_queue.get(timeout=0.01)
                        break
                    except Empty:
                        continue

            if not math.isnan(free_buffer) and not self.terminate:
                with timing.add_time('stack'):
                    rnn_states = torch.stack(rnn_states)
                    num_samples = rnn_states.shape[0]
                    self.device_rnn_states[free_buffer][0:num_samples] = rnn_states

                    for key, x in observations.items():
                        x_stacked = torch.stack(x)
                        with timing.add_time('copy_to_device'):
                            self.device_obs[free_buffer][key][0:num_samples] = x_stacked

                        # TODO: use device_and_type_for_input_tensor
                        # with timing.add_time('obs_to_device'):
                        #     device, dtype = self.actor_critic.device_and_type_for_input_tensor(key)
                        #     observations[key] = x_stacked.to(device).type(dtype)

                self.inference_fast_queue.put((free_buffer, num_samples, requests))

    # noinspection PyProtectedMember
    def _run(self):
        self.device = _init_process(f'inference_batcher_{self.policy_id}_{self.worker_idx}', self.cfg, self.policy_id)
        timing = Timing()

        with timing.timeit('init'):
            # we need it here only to query some info
            actor_critic = create_actor_critic(self.cfg, self.obs_space, self.action_space, timing)
            actor_critic.model_to_device(self.device)

            # init the shared CUDA buffer
            for i in range(self.num_device_buffers):
                if isinstance(self.obs_space, spaces.Dict):
                    for space_name, space in self.obs_space.spaces.items():
                        tensor_type, obs_shape = space.dtype, space.shape
                        if not isinstance(tensor_type, torch.dtype):
                            tensor_type = to_torch_dtype(tensor_type)

                        obs_shape = [self.max_inference_batch] + list(obs_shape)
                        device, _ = actor_critic.device_and_type_for_input_tensor(space_name)
                        self.device_obs[i][space_name] = torch.zeros(obs_shape, dtype=tensor_type, device=device)
                        self.device_obs[i][space_name].share_memory_()
                else:
                    raise Exception('Only Dict observations spaces are supported')

                hidden_size = get_hidden_size(self.cfg)
                hidden_shape = [self.max_inference_batch, hidden_size]
                self.device_rnn_states.append(torch.zeros(hidden_shape, device=self.device))
                self.device_rnn_states[i].share_memory_()

            self.inference_queue.put((TaskType.INIT, (self.device_obs, self.device_rnn_states)))

            del actor_critic
            if self.cfg.device == 'gpu':
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        last_report = time.time()
        last_report_samples = 0
        request_count = deque(maxlen=50)

        # very conservative limit on the minimum number of requests to wait for
        # this will almost guarantee that the system will continue collecting experience
        # at max rate even when 2/3 of workers are stuck for some reason (e.g. doing a long env reset)
        # Although if your workflow involves very lengthy operations that often freeze workers, it can be beneficial
        # to set min_num_requests to 1 (at a cost of potential inefficiency, i.e. policy worker will use very small
        # batches)
        min_num_requests = self.cfg.num_workers // (self.cfg.num_policies * self.cfg.policy_workers_per_policy)
        min_num_requests //= 3
        min_num_requests = max(1, min_num_requests)

        # Again, very conservative timer. Only wait a little bit, then continue operation.
        # Even in the worst case
        wait_for_min_requests = 0.025

        requests = []

        while not self.terminate:
            try:
                while self.stop_experience_collection[self.policy_id]:
                    with self.resume_experience_collection_cv:
                        self.resume_experience_collection_cv.wait(timeout=0.05)

                waiting_started = time.time()
                while len(requests) < min_num_requests and time.time() - waiting_started < wait_for_min_requests:
                    try:
                        with timing.timeit('wait_policy'), timing.add_time('wait_policy_total'):
                            policy_requests = self.policy_queue.get_many(timeout=0.005)
                        requests.extend(policy_requests)
                    except Empty:
                        pass

                with timing.timeit('one_step'), timing.add_time('handle_policy_step'):
                    if self.initialized:
                        if len(requests) > 0:
                            request_count.append(len(requests))
                            self._handle_policy_steps(requests, timing)
                            requests = []

                try:
                    task_type, data = self.task_queue.get_nowait()

                    # task from the task_queue
                    if task_type == TaskType.INIT:
                        self._init()
                    elif task_type == TaskType.TERMINATE:
                        self.inference_fast_queue.put(None)
                        self.terminate = True
                        break

                    self.task_queue.task_done()
                except Empty:
                    pass

                if time.time() - last_report > 3.0 and 'one_step' in timing:
                    timing_stats = dict(wait_policy=timing.wait_policy, step_policy=timing.one_step)
                    samples_since_last_report = self.total_num_samples - last_report_samples

                    stats = memory_stats('policy_worker', self.device)
                    if len(request_count) > 0:
                        stats['avg_request_count'] = np.mean(request_count)

                    self.report_queue.put(dict(
                        timing=timing_stats, samples=samples_since_last_report, policy_id=self.policy_id, stats=stats,
                    ))
                    last_report = time.time()
                    last_report_samples = self.total_num_samples

            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected on worker %d-%d', self.policy_id, self.worker_idx)
                self.terminate = True
            except:
                log.exception('Unknown exception on policy worker')
                self.terminate = True

        time.sleep(0.2)
        log.info('Policy worker avg. requests %.2f, timing: %s', np.mean(request_count), timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.initialized_event.wait()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)
        join_or_kill(self.inference_worker.process)
