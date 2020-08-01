import multiprocessing
import signal
import threading
import time
from collections import deque
from queue import Empty, Queue as RegularQueue

import numpy as np
import psutil
import torch
from torch.multiprocessing import Process as TorchProcess

from algorithms.appo.appo_utils import TaskType, memory_stats, cuda_envvars_for_policy
from algorithms.appo.model import create_actor_critic
from utils.timing import Timing
from utils.utils import AttrDict, log, join_or_kill


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
        self.tensors_individual_transitions = self.shared_buffers.tensors_individual_transitions
        self.policy_versions = shared_buffers.policy_versions
        self.stop_experience_collection = shared_buffers.stop_experience_collection

        self.latest_policy_version = -1
        self.num_policy_updates = 0

        self.total_num_samples = 0

        self.process = TorchProcess(target=self._run, daemon=True)

        self.inference_in_background_thread = True
        if self.inference_in_background_thread:
            self.inference_thread = threading.Thread(target=self._inference_loop)
        else:
            self.inference_thread = None

        self.inference_queue = RegularQueue()

    def _inference_loop(self):
        timing = Timing()

        while not self.terminate:
            self._inference(timing)

        log.info('Policy worker inference thread timing: %s', timing)

    def _inference(self, timing):
        try:
            observations, rnn_states, requests = self.inference_queue.get(timeout=0.1)
        except Empty:
            return

        num_samples = rnn_states.shape[0]

        with timing.add_time('forward'):
            policy_outputs = self.actor_critic(observations, rnn_states)

        for key, output_value in policy_outputs.items():
            policy_outputs[key] = output_value.cpu()

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
            self._enqueue_policy_outputs(requests, output_tensors)

    def start_process(self):
        self.process.start()

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
                for request in requests:
                    actor_idx, split_idx, request_data = request

                    for env_idx, agent_idx, traj_buffer_idx, rollout_step in request_data:
                        index = actor_idx, split_idx, env_idx, agent_idx, traj_buffer_idx, rollout_step
                        dict_of_lists_append(observations, traj_tensors['obs'], index)
                        rnn_states.append(traj_tensors['rnn_states'][index])
                        self.total_num_samples += 1

            with timing.add_time('stack'):
                for key, x in observations.items():
                    observations[key] = torch.stack(x)
                rnn_states = torch.stack(rnn_states)

            with timing.add_time('obs_to_device'):
                for key, x in observations.items():
                    device, dtype = self.actor_critic.device_and_type_for_input_tensor(key)
                    observations[key] = x.to(device).type(dtype)

                rnn_states = rnn_states.to(self.device).float()

            self.inference_queue.put((observations, rnn_states, requests))

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

    def _init_model(self, init_model_data):
        policy_version, state_dict = init_model_data
        self.actor_critic.load_state_dict(state_dict)
        self.shared_model_weights = state_dict
        self.latest_policy_version = policy_version

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

            if self.inference_in_background_thread:
                self.inference_thread.start()

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

        # Again, very conservative timer. Only wait a little bit, then continue operation.
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

                self._update_weights(timing)

                with timing.timeit('one_step'), timing.add_time('handle_policy_step'):
                    if self.initialized:
                        if len(requests) > 0:
                            request_count.append(len(requests))
                            self._handle_policy_steps(requests, timing)
                            requests = []

                if not self.inference_in_background_thread:
                    self._inference(timing)

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


# [2020-07-31 20:53:06,485][05489] Env runner 1, CPU aff. [1], rollouts 800: timing wait_actor: 0.0000, waiting: 1.0178, reset: 17.8580, save_policy_outputs: 0.9685, env_step: 36.4777, overhead: 3.7761, complete_rollouts: 0.0156, enqueue_policy_requests: 0.2282, one_step: 0.0157, work: 43.4498
# [2020-07-31 20:53:06,499][05488] Env runner 0, CPU aff. [0], rollouts 780: timing wait_actor: 0.0000, waiting: 1.0120, reset: 15.0302, save_policy_outputs: 0.9580, env_step: 36.6620, overhead: 3.6411, complete_rollouts: 0.0150, enqueue_policy_requests: 0.2301, one_step: 0.0193, work: 43.4494
# [2020-07-31 20:53:06,750][05487] Policy worker avg. requests 6.60, timing: init: 1.8539, wait_policy_total: 14.7760, wait_policy: 0.0051, handle_policy_step: 33.5766, one_step: 0.0000, deserialize: 1.1792, obs_to_device: 4.4591, stack: 11.8507, forward: 11.1344, postprocess: 4.0673, weight_update: 0.0005
# [2020-07-31 20:53:06,836][05470] GPU learner timing: extract: 0.1849, buffers: 0.0668, batching: 4.7578, buff_ready: 0.2289, tensors_gpu_float: 1.6919, squeeze: 0.0068, prepare: 6.8139, batcher_mem: 4.6625
# [2020-07-31 20:53:07,144][05470] Train loop timing: init: 1.3948, train_wait: 0.4017, epoch_init: 0.0012, minibatch_init: 0.0006, forward_head: 0.4460, bptt_initial: 0.0192, bptt_forward_core: 0.8504, bptt_rnn_states: 0.2146, bptt: 1.1876, tail: 0.2804, vtrace: 0.8958, losses: 0.2586, clip: 6.2688, update: 9.9429, after_optimizer: 0.0869, train: 15.0997
# [2020-07-31 20:53:07,304][05415] Workers joined!
# [2020-07-31 20:53:07,308][05415] Collected {0: 2015232}, FPS: 45337.6
# [2020-07-31 20:53:07,308][05415] Timing: experience: 44.2688
# [2020-07-31 20:53:07,809][05415] Done!


# [2020-07-31 21:38:01,245][01143] Env runner 0, CPU aff. [0], rollouts 800: timing wait_actor: 0.0000, waiting: 1.1049, reset: 14.1496, save_policy_outputs: 0.9768, env_step: 36.4252, overhead: 3.6942, complete_rollouts: 0.0296, enqueue_policy_requests: 0.2016, one_step: 0.0152, work: 43.3196
# [2020-07-31 21:38:01,274][01144] Env runner 1, CPU aff. [1], rollouts 800: timing wait_actor: 0.0000, waiting: 1.1697, reset: 13.6985, save_policy_outputs: 1.0231, env_step: 36.3258, overhead: 3.7354, complete_rollouts: 0.0191, enqueue_policy_requests: 0.1870, one_step: 0.0155, work: 43.2713
# [2020-07-31 21:38:01,516][01142] Policy worker avg. requests 6.90, timing: init: 1.7728, wait_policy_total: 9.6952, wait_policy: 0.0051, handle_policy_step: 6.5976, one_step: 0.0003, deserialize: 1.2105, stack: 4.9503, obs_to_device: 4.5007, forward: 11.2128, postprocess: 4.1313, weight_update: 0.0009
# [2020-07-31 21:38:01,599][01117] GPU learner timing: extract: 0.1895, buffers: 0.0669, batching: 4.7170, buff_ready: 0.2405, tensors_gpu_float: 1.8623, squeeze: 0.0051, prepare: 6.9555, batcher_mem: 4.6175
# [2020-07-31 21:38:01,906][01117] Train loop timing: init: 1.3988, train_wait: 0.3099, epoch_init: 0.0012, minibatch_init: 0.0006, forward_head: 0.4520, bptt_initial: 0.0181, bptt_forward_core: 0.8651, bptt_rnn_states: 0.2231, bptt: 1.2106, tail: 0.2881, vtrace: 0.9162, losses: 0.2348, clip: 6.2798, update: 10.1031, after_optimizer: 0.0985, train: 15.2374
# [2020-07-31 21:38:02,075][01061] Workers joined!
# [2020-07-31 21:38:02,086][01061] Collected {0: 2015232}, FPS: 45373.6
# [2020-07-31 21:38:02,086][01061] Timing: experience: 44.2337
# [2020-07-31 21:38:02,587][01061] Done!
