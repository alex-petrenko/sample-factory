import select
import time
from queue import Empty

import torch
from torch.multiprocessing import Process as TorchProcess, Event

from algorithms.appo.appo_utils import TaskType, dict_of_lists_append, memory_stats, cuda_envvars
from algorithms.appo.model import ActorCritic
from utils.timing import Timing
from utils.utils import AttrDict, log


class PolicyWorker:
    def __init__(
        self, worker_idx, policy_id, cfg, obs_space, action_space, policy_queue, actor_queues,
        report_queue, policy_worker_queues,
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

        self.input_tensors = dict()  # memshared input tensors, main way to receive data from actors
        self.output_tensors = dict()

        self.policy_queue = policy_queue
        self.actor_queues = actor_queues
        self.report_queue = report_queue

        # queue other components use to talk to this particular worker
        self.task_queue = policy_worker_queues[policy_id][worker_idx]

        # queues for all other policy workers, in case we need to talk to them (e.g. send initial tensor buffers)
        self.policy_worker_queues = policy_worker_queues

        self.initialized = False
        self.terminate = False

        self.latest_policy_version = 0

        self.requests = dict()

        self.total_num_samples = 0

        if self.cfg.benchmark:
            self.max_requests_allowed = 1e9  # unlimited from the start
        else:
            self.max_requests_allowed = 10

        self.process = TorchProcess(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _init(self):
        log.info('GPU worker %d-%d initialized', self.policy_id, self.worker_idx)
        self.initialized = True

    def _store_policy_step_request(self, request):
        worker_idx, split_idx, _ = request
        self.requests[(worker_idx, split_idx)] = request

    def _filter_requests(self):
        requests, to_remove = [], []

        for worker_split, request in self.requests.items():
            requests.append(request)
            to_remove.append(worker_split)

            if len(requests) > self.max_requests_allowed:
                # this is a simple heuristic to allow the policy worker to ramp up gradually
                # and avoid using too much CUDA memory right from the start
                break

        for worker_split in to_remove:
            del self.requests[worker_split]

        return requests

    def _handle_policy_steps(self, requests, timing):
        self.max_requests_allowed += 1

        with timing.add_time('deserialize'):
            observations = AttrDict()
            rnn_states = []
            request_order = []

            for request in requests:
                actor_idx, split_idx, request_data = request

                for env_idx, agent_idx, rollout_step in request_data:
                    tensors_dict_key = (actor_idx, split_idx, env_idx, agent_idx)
                    input_tensors = self.input_tensors[tensors_dict_key]
                    dict_of_lists_append(observations, input_tensors['obs'])
                    rnn_states.append(input_tensors['rnn_states'])
                    request_order.append(tensors_dict_key)
                    self.total_num_samples += 1

        with torch.no_grad():
            with timing.add_time('to_device'):
                for key, x in observations.items():
                    observations[key] = torch.stack(x).to(self.device).float()

                rnn_states = torch.stack(rnn_states).to(self.device).float()
                num_samples = rnn_states.shape[0]

            with timing.add_time('forward'):
                policy_outputs = self.actor_critic(observations, rnn_states)
                policy_outputs.policy_version = torch.empty([num_samples]).fill_(self.latest_policy_version)

            for key, value in policy_outputs.items():
                policy_outputs[key] = value.cpu()

            # concat all tensors into a single tensor for performance
            output_tensors, tensor_sizes = [], []
            tensor_names = sorted(tuple(policy_outputs.keys()))
            for key in tensor_names:
                value = policy_outputs[key].float()
                if len(value.shape) == 1:
                    value.unsqueeze_(dim=1)
                output_tensors.append(value)
                tensor_sizes.append(value.shape[-1])

            output_tensors = torch.cat(output_tensors, dim=1)

            with timing.add_time('postprocess'):
                self._enqueue_policy_outputs(request_order, output_tensors, tensor_names, tensor_sizes)

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

    def _initialize_shared_output_tensors(self, init_tensors_request, actor_idx):
        msg = (TaskType.INIT_TENSORS, init_tensors_request)

        # we're sending the tensors to the actor they belong to
        self.actor_queues[actor_idx].put(msg)

        # and to all other policy workers that should all share same memory
        for policy_id in range(self.cfg.num_policies):
            for policy_worker_idx in range(self.cfg.policy_workers_per_policy):
                if policy_id == self.policy_id and self.worker_idx == policy_worker_idx:
                    # don't send this message to ourselves
                    continue

                self.policy_worker_queues[policy_id][policy_worker_idx].put(msg)

    def _enqueue_policy_outputs(self, request_order, output_tensors, tensor_names, tensor_sizes):
        output_idx = 0
        outputs_ready = set()

        for actor_idx, split_idx, env_idx, agent_idx in request_order:
            tensors_dict_key = actor_idx, split_idx, env_idx, agent_idx

            if tensors_dict_key in self.output_tensors:
                self.output_tensors[tensors_dict_key].copy_(output_tensors[output_idx])
            else:
                shared_output_tensors = output_tensors[output_idx].clone().detach()
                shared_output_tensors.share_memory_()
                self.output_tensors[tensors_dict_key] = shared_output_tensors

                log.debug('Sending ouput tensors for policy %d to %r', self.policy_id, tensors_dict_key)
                init_tensors_request = dict(
                    actor_idx=actor_idx, split_idx=split_idx, env_idx=env_idx, agent_idx=agent_idx,
                    tensors=shared_output_tensors,
                    tensor_names=tensor_names, tensor_sizes=tensor_sizes,
                    init_output_tensors=True,
                )
                self._initialize_shared_output_tensors(init_tensors_request, actor_idx)

            output_idx += 1

            outputs_ready.add((actor_idx, split_idx))

        for actor_idx, split_idx in outputs_ready:
            advance_rollout_request = dict(split_idx=split_idx, policy_id=self.policy_id)
            self.actor_queues[actor_idx].put((TaskType.ROLLOUT_STEP, advance_rollout_request))

    def _init_input_tensors(self, orig_data):
        data = AttrDict(orig_data)
        assert data.policy_id == self.policy_id
        assert data.policy_worker_idx == self.worker_idx

        worker_idx, split_idx = data.worker_idx, data.split_idx
        log.debug(
            'Policy worker %d-%d initializing input tensors from %d %d',
            self.policy_id, self.worker_idx, worker_idx, split_idx,
        )

        for key, tensors in data.tensors.items():
            env_idx, agent_idx = key
            tensors_dict_key = (worker_idx, split_idx, env_idx, agent_idx)
            self.input_tensors[tensors_dict_key] = tensors

    def _init_output_tensors(self, orig_data):
        data = AttrDict(orig_data)
        tensors_dict_key = (data.actor_idx, data.split_idx, data.env_idx, data.agent_idx)
        log.debug(
            'Policy worker %d-%d initializing output tensors for %d %d',
            self.policy_id, self.worker_idx, data.actor_idx, data.split_idx,
        )
        self.output_tensors[tensors_dict_key] = data.tensors

    # noinspection PyProtectedMember
    def _run(self):
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

        queues = [self.task_queue._reader, self.policy_queue._reader]
        queues_by_handle = dict()
        queues_by_handle[self.policy_queue._reader._handle] = self.policy_queue
        queues_by_handle[self.task_queue._reader._handle] = self.task_queue

        last_report = last_cache_cleanup = time.time()
        last_report_samples = 0

        while not self.terminate:
            try:
                with timing.add_time('gpu_waiting'), timing.timeit('wait_policy'):
                    ready, _, _ = select.select(queues, [], [])

                with timing.add_time('work'):
                    for readable_queue in ready:
                        q = queues_by_handle[readable_queue._handle]

                        with timing.add_time('loop'):
                            while True:
                                try:
                                    task_type, data = q.get_nowait()

                                    if task_type == TaskType.POLICY_STEP:
                                        self._store_policy_step_request(data)
                                    else:
                                        # task from the task_queue
                                        if task_type == TaskType.INIT:
                                            self._init()
                                        elif task_type == TaskType.TERMINATE:
                                            self.terminate = True
                                            break
                                        elif task_type == TaskType.INIT_TENSORS:
                                            if 'init_output_tensors' in data:
                                                self._init_output_tensors(data)
                                            else:
                                                self._init_input_tensors(data)
                                        elif task_type == TaskType.UPDATE_WEIGHTS:
                                            with timing.timeit('updates'):
                                                self._update_weights(data, timing)

                                        self.task_queue.task_done()

                                except Empty:
                                    break

                    with timing.timeit('one_step'), timing.add_time('handle_policy_step'):
                        if self.initialized:
                            requests_to_process = self._filter_requests()
                            if len(requests_to_process) > 0:
                                self._handle_policy_steps(requests_to_process, timing)

                    if time.time() - last_report > 3.0 and 'one_step' in timing:
                        timing_stats = dict(wait_policy=timing.wait_policy, step_policy=timing.one_step)
                        samples_since_last_report = self.total_num_samples - last_report_samples

                        stats = memory_stats('policy_worker', self.device)

                        self.report_queue.put(dict(
                            timing=timing_stats, samples=samples_since_last_report, policy_id=self.policy_id, stats=stats,
                        ))
                        last_report = time.time()
                        last_report_samples = self.total_num_samples

                    if time.time() - last_cache_cleanup > 30.0 or (not self.cfg.benchmark and self.total_num_samples < 1000):
                        torch.cuda.empty_cache()
                        last_cache_cleanup = time.time()

            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected on worker %d-%d', self.policy_id, self.worker_idx)
                self.terminate = True

        log.info('Policy worker timing: %s', timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.join()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        self.process.join(timeout=5)
