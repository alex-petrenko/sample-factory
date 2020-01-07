import random
import select
import time
from queue import Empty

import torch
from torch.multiprocessing import Process as TorchProcess

from algorithms.appo.appo_utils import TaskType, dict_of_lists_append, device_for_policy, set_step_data
from algorithms.appo.model import ActorCritic
from algorithms.utils.algo_utils import EPS
from utils.timing import Timing
from utils.utils import AttrDict, log


class PolicyWorker:
    def __init__(
            self, worker_idx, policy_id, cfg, obs_space, action_space, policy_queue, actor_queues,
            weight_queue, report_queue,
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

        self.task_queue = policy_queue
        self.actor_queues = actor_queues
        self.weight_queue = weight_queue
        self.report_queue = report_queue

        self.terminate = False

        self.latest_policy_version = 0

        self.requests = dict()
        self.workers_paused = set()
        self.too_many_workers = False
        self.last_speed_adjustment = time.time()

        self.total_num_samples = 0

        self.process = TorchProcess(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _init(self):
        log.info('GPU worker %d initialized', self.worker_idx)

    def _terminate(self):
        pass

    def _store_policy_step_request(self, request):
        worker_idx = request['worker_idx']
        split_idx = request['split_idx']
        self.requests[(worker_idx, split_idx)] = request

    def _filter_requests(self):
        requests, to_remove = [], []
        for worker_split, request in self.requests.items():
            if worker_split not in self.workers_paused:
                requests.append(request)
                to_remove.append(worker_split)

        for worker_split in to_remove:
            del self.requests[worker_split]

        return requests

    def _handle_policy_steps(self, timing):
        requests = self._filter_requests()

        # log.info('Num pending requests: %d', len(requests))
        if len(requests) <= 0:
            return

        with timing.add_time('deserialize'):
            observations = AttrDict()
            rnn_states = []
            request_order = []

            for request in requests:
                actor_idx = request['worker_idx']
                split_idx = request['split_idx']
                request_data = request['policy_inputs']

                rollout_step = -1
                for env_idx, agent_idx, rollout_step in request_data:
                    tensors_dict_key = (actor_idx, split_idx, env_idx, agent_idx)
                    input_tensors = self.input_tensors[tensors_dict_key]
                    dict_of_lists_append(observations, input_tensors['obs'])
                    rnn_states.append(input_tensors['rnn_states'])
                    request_order.append(tensors_dict_key)
                    self.total_num_samples += 1

                if rollout_step >= self.cfg.rollout - 1:
                    if self.cfg.sync_mode or self.too_many_workers:
                        self.workers_paused.add((actor_idx, split_idx))

                        if self.too_many_workers:
                            log.warning(
                                'Paused worker (%d, %d), total %d (%r)',
                                actor_idx, split_idx, len(self.workers_paused), self.workers_paused,
                            )
                            self.too_many_workers = False

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
            output_tensors, tensor_sizes = [], []
            tensor_names = sorted(tuple(policy_outputs.keys()))
            for key in tensor_names:
                value = policy_outputs[key].float()
                if len(value.shape) == 1:
                    value = torch.unsqueeze(value, dim=1)
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

        # adjust experience collection rate
        if time.time() - self.last_speed_adjustment > 30.0:
            if discarding_rate <= 0:
                if self.workers_paused:
                    worker_idx = self.workers_paused.pop()
                    log.warning(
                        'Resume experience collection on worker %r (paused %d: %r)',
                        worker_idx, len(self.workers_paused), self.workers_paused,
                    )

                    self.last_speed_adjustment = time.time()
                    self.too_many_workers = False

            elif discarding_rate > EPS:
                # learner isn't fast enough to process all experience - disable a single worker to reduce
                # collection rate
                # We will pause the worker when the entire rollout is collected
                log.warning('Worker is requested to pause after the end of the next rollout')
                self.last_speed_adjustment = time.time()
                self.too_many_workers = True

        if self.cfg.sync_mode:
            self.workers_paused.clear()

        log.info(
            'Updated weights on worker %d, policy_version %d (%.5f)',
            self.worker_idx, policy_version, timing.weight_update,
        )

    def _enqueue_policy_outputs(self, request_order, output_tensors, tensor_names, tensor_sizes):
        output_idx = 0
        outputs_ready = set()

        for actor_idx, split_idx, env_idx, agent_idx in request_order:
            tensors_dict_key = actor_idx, split_idx, env_idx, agent_idx

            if tensors_dict_key in self.output_tensors:
                self.output_tensors[tensors_dict_key].copy_(output_tensors[output_idx])
            else:
                self.output_tensors[tensors_dict_key] = output_tensors[output_idx].clone().detach()
                self.output_tensors[tensors_dict_key].share_memory_()

                log.debug('Sending ouput tensors for policy %d to %r', self.policy_id, tensors_dict_key)
                init_tensors_request = dict(
                    actor_idx=actor_idx, split_idx=split_idx, env_idx=env_idx, agent_idx=agent_idx,
                    policy_id=self.policy_id,
                    tensors=self.output_tensors[tensors_dict_key],
                    tensor_names=tensor_names, tensor_sizes=tensor_sizes,
                    init_output_tensors=True,
                )
                self.actor_queues[actor_idx].put((TaskType.INIT_TENSORS, init_tensors_request))

            output_idx += 1

            outputs_ready.add((actor_idx, split_idx))

        for actor_idx, split_idx in outputs_ready:
            advance_rollout_request = dict(split_idx=split_idx, policy_id=self.policy_id)
            self.actor_queues[actor_idx].put((TaskType.ROLLOUT_STEP, advance_rollout_request))

    def _init_input_tensors(self, data):
        data = AttrDict(data)
        worker_idx, split_idx = data.worker_idx, data.split_idx
        log.debug('Policy worker %d initializing input tensors from %d %d', self.policy_id, worker_idx, split_idx)

        for key, tensors in data.tensors.items():
            env_idx, agent_idx = key
            tensors_dict_key = (worker_idx, split_idx, env_idx, agent_idx)
            self.input_tensors[tensors_dict_key] = tensors

    def _init_output_tensors(self, data):
        data = AttrDict(data)
        tensors_dict_key = (data.actor_idx, data.split_idx, data.env_idx, data.agent_idx)
        assert tensors_dict_key not in self.output_tensors
        self.output_tensors[tensors_dict_key] = data.tensors

    # noinspection PyProtectedMember
    def _run(self):
        timing = Timing()

        with timing.timeit('init'):
            # initialize the Torch modules
            log.info('Initializing model on the policy worker %d...', self.worker_idx)

            torch.set_num_threads(1)

            self.device = device_for_policy(self.policy_id)
            self.actor_critic = ActorCritic(self.obs_space, self.action_space, self.cfg)
            self.actor_critic.to(self.device)

            log.info('Initialized model on the policy worker %d!', self.worker_idx)

        queues = [self.task_queue._reader, self.weight_queue._reader]
        queues_by_handle = dict()
        queues_by_handle[self.task_queue._reader._handle] = self.task_queue
        queues_by_handle[self.weight_queue._reader._handle] = self.weight_queue

        last_report = time.time()
        last_report_samples = 0

        while not self.terminate:
            with timing.add_time('gpu_waiting'), timing.timeit('wait_policy'):
                ready, _, _ = select.select(queues, [], [])

            with timing.add_time('work'):
                for readable_queue in ready:
                    q = queues_by_handle[readable_queue._handle]

                    with timing.add_time('loop'):
                        while True:
                            try:
                                task_type, data = q.get_nowait()

                                if task_type == TaskType.INIT:
                                    self._init()
                                elif task_type == TaskType.TERMINATE:
                                    self._terminate()
                                    self.terminate = True
                                    break
                                elif task_type == TaskType.INIT_TENSORS:
                                    if 'init_output_tensors' in data:
                                        self._init_output_tensors(data)
                                    else:
                                        self._init_input_tensors(data)
                                elif task_type == TaskType.POLICY_STEP:
                                    self._store_policy_step_request(data)
                                elif task_type == TaskType.UPDATE_WEIGHTS:
                                    with timing.timeit('updates'):
                                        self._update_weights(data, timing)

                            except Empty:
                                break

                with timing.timeit('one_step'), timing.add_time('handle_policy_step'):
                    self._handle_policy_steps(timing)

                if time.time() - last_report > 1.0 and 'one_step' in timing:
                    timing_stats = dict(wait_policy=timing.wait_policy, step_policy=timing.one_step)
                    samples_since_last_report = self.total_num_samples - last_report_samples
                    self.report_queue.put(dict(
                        timing=timing_stats, samples=samples_since_last_report, policy_id=self.policy_id,
                    ))
                    last_report = time.time()
                    last_report_samples = self.total_num_samples

        log.info('Gpu worker timing: %s', timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.put((TaskType.EMPTY, None))
        while self.task_queue.qsize() > 0:
            time.sleep(0.01)

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        self.process.join(timeout=5)
