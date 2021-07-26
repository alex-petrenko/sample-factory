import multiprocessing
import os
import signal
import time
from collections import deque
from queue import Empty

import numpy as np
import psutil
import torch
from torch.multiprocessing import Process as TorchProcess

from sample_factory.algorithms.appo.appo_utils import TaskType, memory_stats, cuda_envvars_for_policy
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import AttrDict, log, join_or_kill


def dict_of_lists_append(dict_of_lists, new_data, index):
    for key, x in new_data.items():
        if key in dict_of_lists:
            dict_of_lists[key].append(x[index])
        else:
            dict_of_lists[key] = [x[index]]


class PolicyWorker:
    def __init__(
        self, worker_idx, policy_id, cfg, obs_space, action_space, shared_buffers, policy_queue, actor_queues,
        report_queue, task_queue, policy_lock, resume_experience_collection_cv
    ):
        log.info('Initializing policy worker %d for policy %d', worker_idx, policy_id)

        self.worker_idx = worker_idx
        self.policy_id = policy_id
        self.cfg = cfg

        self.obs_space = obs_space
        self.action_space = action_space

        self.device = None
        self.actor_critic = None
        self.shared_model_weights = None
        self.policy_lock = policy_lock
        self.resume_experience_collection_cv = resume_experience_collection_cv

        self.policy_queue = policy_queue
        self.actor_queues = actor_queues
        self.report_queue = report_queue

        # queue other components use to talk to this particular worker
        self.task_queue = task_queue

        self.initialized = False
        self.terminate = False
        self.initialized_event = multiprocessing.Event()
        self.initialized_event.clear()

        self.shared_buffers = shared_buffers
        self.traj_tensors = self.shared_buffers.tensors
        self.policy_outputs = self.shared_buffers.policy_output_tensors

        self.latest_policy_version = -1
        self.num_policy_updates = 0

        self.requests = []

        self.total_num_samples = 0

        self.process = TorchProcess(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _init(self):
        log.info('Policy worker %d-%d initialized', self.policy_id, self.worker_idx)
        self.initialized = True
        self.initialized_event.set()

    def _handle_policy_steps(self, timing):
        with torch.no_grad():
            with timing.add_time('deserialize'):
                indices = []
                for request in self.requests:
                    actor_idx, split_idx, request_data = request

                    for env_idx, agent_idx, traj_buffer_idx, rollout_step in request_data:
                        index = [traj_buffer_idx, rollout_step]
                        indices.append(index)
                        self.total_num_samples += 1

                indices = tuple(np.array(indices).T)
                observations = self.traj_tensors['obs'].index(indices)
                rnn_states = self.traj_tensors['rnn_states'][indices]

            with timing.add_time('stack'):
                for key, x in observations.items():
                    observations[key] = torch.from_numpy(x)
                rnn_states = torch.from_numpy(rnn_states)
                num_samples = rnn_states.shape[0]

            with timing.add_time('obs_to_device'):
                for key, x in observations.items():
                    device, dtype = self.actor_critic.device_and_type_for_input_tensor(key)
                    observations[key] = x.to(device).type(dtype)
                rnn_states = rnn_states.to(self.device).float()

            with timing.add_time('forward'):
                policy_outputs = self.actor_critic(observations, rnn_states)

            with timing.add_time('to_cpu'):
                for key, output_value in policy_outputs.items():
                    policy_outputs[key] = output_value.cpu()

            with timing.add_time('format_outputs'):
                policy_outputs.policy_version = torch.empty([num_samples]).fill_(self.latest_policy_version)

                # concat all tensors into a single tensor for performance
                output_tensors = []
                for policy_output in self.shared_buffers.policy_outputs:
                    tensor_name = policy_output.name
                    output_value = policy_outputs[tensor_name].float()
                    if len(output_value.shape) == 1:
                        output_value.unsqueeze_(dim=1)
                    output_tensors.append(output_value)

                output_tensors = torch.cat(output_tensors, dim=1)

            with timing.add_time('postprocess'):
                self._enqueue_policy_outputs(self.requests, output_tensors)

        self.requests = []

    def _enqueue_policy_outputs(self, requests, output_tensors):
        outputs_ready = set()

        output_indices = []
        for request in requests:
            actor_idx, split_idx, request_data = request
            for env_idx, agent_idx, traj_buffer_idx, rollout_step in request_data:
                output_indices.append([actor_idx, split_idx, env_idx, agent_idx])

            outputs_ready.add((actor_idx, split_idx))

        output_indices = tuple(np.array(output_indices).T)
        self.policy_outputs[output_indices] = output_tensors.numpy()

        for actor_idx, split_idx in outputs_ready:
            advance_rollout_request = dict(split_idx=split_idx, policy_id=self.policy_id)
            self.actor_queues[actor_idx].put((TaskType.ROLLOUT_STEP, advance_rollout_request))

    def _init_model(self, init_model_data):
        policy_version, state_dict = init_model_data
        self.actor_critic.load_state_dict(state_dict)
        self.shared_model_weights = state_dict
        self.latest_policy_version = policy_version

    def _update_weights(self, timing):
        learner_policy_version = self.shared_buffers.policy_versions[self.policy_id].item()
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

    # noinspection PyProtectedMember
    def _run(self):
        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        psutil.Process().nice(min(self.cfg.default_niceness + 2, 20))

        cuda_envvars_for_policy(self.policy_id, 'inference')
        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        with timing.timeit('init'):
            # initialize the Torch modules
            log.info('Initializing model on the policy worker %d-%d...', self.policy_id, self.worker_idx)
            log.info(f'POLICY worker {self.policy_id}-{self.worker_idx}\tpid {os.getpid()}\tparent {os.getppid()}')

            torch.set_num_threads(1)

            if self.cfg.device == 'gpu':
                # we should already see only one CUDA device, because of env vars
                assert torch.cuda.device_count() == 1
                self.device = torch.device('cuda', index=0)
            else:
                self.device = torch.device('cpu')

            self.actor_critic = create_actor_critic(self.cfg, self.obs_space, self.action_space, timing)
            self.actor_critic.model_to_device(self.device)
            for p in self.actor_critic.parameters():
                p.requires_grad = False  # we don't train anything here

            log.info('Initialized model on the policy worker %d-%d!', self.policy_id, self.worker_idx)

        last_report = last_cache_cleanup = time.time()
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
        log.info('Min num requests: %d', min_num_requests)

        # Again, very conservative timer. Only wait a little bit, then continue operation.
        wait_for_min_requests = 0.025

        while not self.terminate:
            try:
                while self.shared_buffers.stop_experience_collection[self.policy_id]:
                    with self.resume_experience_collection_cv:
                        self.resume_experience_collection_cv.wait(timeout=0.05)

                waiting_started = time.time()
                while len(self.requests) < min_num_requests and time.time() - waiting_started < wait_for_min_requests:
                    try:
                        with timing.timeit('wait_policy'), timing.add_time('wait_policy_total'):
                            policy_requests = self.policy_queue.get_many(timeout=0.005)
                        self.requests.extend(policy_requests)
                    except Empty:
                        pass

                self._update_weights(timing)

                with timing.timeit('one_step'), timing.add_time('handle_policy_step'):
                    if self.initialized:
                        if len(self.requests) > 0:
                            request_count.append(len(self.requests))
                            self._handle_policy_steps(timing)

                try:
                    task_type, data = self.task_queue.get_nowait()

                    # task from the task_queue
                    if task_type == TaskType.INIT:
                        self._init()
                    elif task_type == TaskType.TERMINATE:
                        self.terminate = True
                        break
                    elif task_type == TaskType.INIT_MODEL:
                        self._init_model(data)

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

                if time.time() - last_cache_cleanup > 300.0 or (not self.cfg.benchmark and self.total_num_samples < 1000):
                    if self.cfg.device == 'gpu':
                        torch.cuda.empty_cache()
                    last_cache_cleanup = time.time()

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
