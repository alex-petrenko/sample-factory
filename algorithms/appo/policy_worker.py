import signal
import time
from collections import deque
from queue import Empty

import numpy as np
import torch
from torch.multiprocessing import Process as TorchProcess

from algorithms.appo.appo_utils import TaskType, memory_stats, cuda_envvars
from algorithms.appo.model import ActorCritic
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
            self, worker_idx, policy_id, cfg, obs_space, action_space, traj_buffers, policy_queue, actor_queues,
            report_queue, task_queue,
    ):
        log.info('Initializing GPU worker %d for policy %d', worker_idx, policy_id)

        self.worker_idx = worker_idx
        self.policy_id = policy_id
        self.cfg = cfg

        self.obs_space = obs_space
        self.action_space = action_space

        self.device = None
        self.actor_critic = None
        self.shared_model = None

        self.policy_queue = policy_queue
        self.actor_queues = actor_queues
        self.report_queue = report_queue

        # queue other components use to talk to this particular worker
        self.task_queue = task_queue

        self.initialized = False
        self.terminate = False

        self.traj_buffers = traj_buffers
        self.tensors_individual_transitions = self.traj_buffers.tensors_individual_transitions

        self.latest_policy_version = 0

        self.requests = []

        self.total_num_samples = 0

        self.process = TorchProcess(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _init(self):
        log.info('GPU worker %d-%d initialized', self.policy_id, self.worker_idx)
        self.initialized = True

    def _filter_requests(self):
        requests = self.requests
        self.requests = []
        return requests

    def _handle_policy_steps(self, timing):
        with timing.add_time('deserialize'):
            observations = AttrDict()
            rnn_states = []

            traj_tensors = self.traj_buffers.tensors_individual_transitions
            for request in self.requests:
                actor_idx, split_idx, request_data = request

                for env_idx, agent_idx, traj_buffer_idx, rollout_step in request_data:
                    index = actor_idx, split_idx, env_idx, agent_idx, traj_buffer_idx, rollout_step
                    dict_of_lists_append(observations, traj_tensors['obs'], index)
                    rnn_states.append(traj_tensors['rnn_states'][index])
                    self.total_num_samples += 1

        with torch.no_grad():
            with timing.add_time('to_device'):
                for key, x in observations.items():
                    observations[key] = torch.stack(x).to(self.device).float()

                rnn_states = torch.stack(rnn_states).to(self.device).float()
                num_samples = rnn_states.shape[0]

            with timing.add_time('forward'):
                policy_outputs = self.actor_critic(observations, rnn_states)

            for key, value in policy_outputs.items():
                policy_outputs[key] = value.cpu()

            policy_outputs.policy_version = torch.empty([num_samples]).fill_(self.latest_policy_version)

            # concat all tensors into a single tensor for performance
            output_tensors = []
            for policy_output in self.traj_buffers.policy_outputs:
                tensor_name = policy_output.name
                value = policy_outputs[tensor_name].float()
                if len(value.shape) == 1:
                    value.unsqueeze_(dim=1)
                output_tensors.append(value)

            output_tensors = torch.cat(output_tensors, dim=1)

            with timing.add_time('postprocess'):
                self._enqueue_policy_outputs(self.requests, output_tensors)

        self.requests = []

    def _update_weights(self, weight_update, timing):
        if weight_update is None:
            return

        with timing.timeit('weight_update'):
            policy_version, state_dict, discarding_rate = weight_update
            self.actor_critic.load_state_dict(state_dict)
            self.latest_policy_version = policy_version

        log.info(
            'Updated weights on worker %d-%d, policy_version %d (%.5f)',
            self.policy_id, self.worker_idx, policy_version, timing.weight_update,
        )

    def _enqueue_policy_outputs(self, requests, output_tensors):
        output_idx = 0
        outputs_ready = set()

        policy_outputs = self.traj_buffers.policy_output_tensors

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

    # noinspection PyProtectedMember
    def _run(self):
        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        cuda_envvars(self.policy_id)
        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        with timing.timeit('init'):
            # initialize the Torch modules
            log.info('Initializing model on the policy worker %d-%d...', self.policy_id, self.worker_idx)

            torch.set_num_threads(1)

            # we should already see only one CUDA device, because of env vars
            assert torch.cuda.device_count() == 1
            self.device = torch.device('cuda', index=0)
            self.actor_critic = ActorCritic(self.obs_space, self.action_space, self.cfg)
            self.actor_critic.to(self.device)

            log.info('Initialized model on the policy worker %d-%d!', self.policy_id, self.worker_idx)

        last_report = last_cache_cleanup = time.time()
        last_report_samples = 0
        request_count = deque(maxlen=20)

        while not self.terminate:
            try:
                try:
                    with timing.timeit('wait_policy'), timing.add_time('wait_policy_total'):
                        policy_requests = self.policy_queue.get_many(timeout=0.01)
                    self.requests.extend(policy_requests)
                except Empty:
                    pass

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
                    elif task_type == TaskType.UPDATE_WEIGHTS:
                        with timing.timeit('updates'):
                            self._update_weights(data, timing)

                    self.task_queue.task_done()
                except Empty:
                    pass

                if time.time() - last_report > 3.0 and 'one_step' in timing and len(request_count) > 0:
                    timing_stats = dict(wait_policy=timing.wait_policy, step_policy=timing.one_step)
                    samples_since_last_report = self.total_num_samples - last_report_samples

                    stats = memory_stats('policy_worker', self.device)
                    stats['avg_request_count'] = np.mean(request_count)

                    self.report_queue.put(dict(
                        timing=timing_stats, samples=samples_since_last_report, policy_id=self.policy_id, stats=stats,
                    ))
                    last_report = time.time()
                    last_report_samples = self.total_num_samples

                if time.time() - last_cache_cleanup > 300.0 or (not self.cfg.benchmark and self.total_num_samples < 1000):
                    torch.cuda.empty_cache()
                    last_cache_cleanup = time.time()

            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected on worker %d-%d', self.policy_id, self.worker_idx)
                self.terminate = True
            except:
                log.exception('Unknown exception on policy worker')
                self.terminate = True

        log.info('Policy worker timing: %s', timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.join()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)
