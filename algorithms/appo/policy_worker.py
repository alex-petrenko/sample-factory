import select
from queue import Empty

import numpy as np
import ray.pyarrow_files.pyarrow as pa
import torch
from ray.pyarrow_files.pyarrow import plasma
from torch.multiprocessing import Process

from algorithms.appo.appo_utils import TaskType, dict_of_lists_append
from algorithms.appo.model import ActorCritic
from utils.timing import Timing
from utils.utils import AttrDict, log


class PolicyWorker:
    def __init__(
            self, worker_idx, policy_id, cfg, obs_space, action_space, plasma_store_name, policy_queue, actor_queues,
            weight_queue,
    ):
        log.info('Initializing GPU worker %d for policy %d', worker_idx, policy_id)

        self.worker_idx = worker_idx
        self.policy_id = policy_id
        self.cfg = cfg

        self.obs_space = obs_space
        self.action_space = action_space

        self.plasma_store_name = plasma_store_name
        self.plasma_client = None
        self.serialization_context = None

        self.device = None
        self.actor_critic = None
        self.shared_model = None

        self.task_queue = policy_queue
        self.actor_queues = actor_queues
        self.weight_queue = weight_queue

        self.terminate = False

        self.latest_policy_version = 0

        self.requests = dict()
        self.workers_paused = set()

        self.process = Process(target=self._run, daemon=True)

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

        with timing.add_time('policy_step'):
            with timing.add_time('deserialize'):
                observations = AttrDict()
                rnn_states = []
                num_obs_per_actor = []

                for request in requests:
                    actor_idx = request['worker_idx']
                    split_idx = request['split_idx']

                    request_data = self.plasma_client.get(
                        request['policy_inputs'], -1, serialization_context=self.serialization_context,
                    )

                    num_inputs, rollout_step, policy_inputs = request_data
                    # log.info('A:%d  S:%d  num:%d  roll:%d  policy:%d', actor_idx, split_idx, num_inputs, rollout_step, self.latest_policy_version)

                    dict_of_lists_append(observations, policy_inputs['obs'])
                    rnn_states.append(policy_inputs['rnn_states'])
                    num_obs_per_actor.append((actor_idx, split_idx, num_inputs))

                    if self.cfg.sync_mode:
                        if rollout_step >= self.cfg.rollout - 1:
                            self.workers_paused.add((actor_idx, split_idx))

            with torch.no_grad():
                with timing.add_time('to_device'):
                    for key, x in observations.items():
                        observations[key] = torch.from_numpy(np.concatenate(x)).to(self.device).float()

                    rnn_states = np.concatenate(rnn_states)
                    rnn_states = torch.from_numpy(rnn_states).to(self.device).float()

                with timing.add_time('forward'):
                    policy_outputs = self.actor_critic(observations, rnn_states)

                with timing.add_time('postprocess'):
                    self._enqueue_policy_outputs(
                        num_obs_per_actor, policy_outputs, self.serialization_context, self.plasma_client,
                    )

    def _update_weights(self, weight_update, timing):
        if weight_update is None:
            return

        with timing.timeit('weight_update'):
            policy_version, state_dict = weight_update
            self.actor_critic.load_state_dict(state_dict)
            self.latest_policy_version = policy_version

        if self.cfg.sync_mode:
            self.workers_paused.clear()

        log.info(
            'Updated weights on worker %d, policy_version %d (%.5f)',
            self.worker_idx, policy_version, timing.weight_update,
        )

    def _enqueue_policy_outputs(self, num_obs_per_actor, policy_outputs, serialization_context, plasma_client):
        for key, value in policy_outputs.items():
            policy_outputs[key] = value.cpu().numpy()

        output_idx = 0
        for actor_index, split_idx, num_obs in num_obs_per_actor:
            outputs = dict()
            for key, value in policy_outputs.items():
                outputs[key] = value[output_idx:output_idx + num_obs]

            outputs = plasma_client.put(
                outputs, None, serialization_context=serialization_context,
            )

            advance_rollout_request = dict(
                split_idx=split_idx, policy_id=self.policy_id, outputs=outputs,
                policy_version=self.latest_policy_version,
            )
            self.actor_queues[actor_index].put((TaskType.ROLLOUT_STEP, advance_rollout_request))
            output_idx += num_obs

    def _run(self):
        timing = Timing()

        with timing.timeit('init'):
            self.plasma_client = plasma.connect(self.plasma_store_name)
            self.serialization_context = pa.default_serialization_context()

            # initialize the Torch modules
            log.info('Initializing model on the policy worker %d...', self.worker_idx)

            torch.set_num_threads(1)
            self.device = torch.device('cuda')
            self.actor_critic = ActorCritic(self.obs_space, self.action_space, self.cfg)
            self.actor_critic.to(self.device)

            log.info('Initialized model on the policy worker %d!', self.worker_idx)

        queues = [self.task_queue._reader, self.weight_queue._reader]
        queues_by_handle = dict()
        queues_by_handle[self.task_queue._reader._handle] = self.task_queue
        queues_by_handle[self.weight_queue._reader._handle] = self.weight_queue

        while not self.terminate:
            with timing.add_time('gpu_waiting'):
                ready, _, _ = select.select(queues, [], [])

            with timing.add_time('work'):
                for readable_queue in ready:
                    q = queues_by_handle[readable_queue._handle]

                    while True:
                        try:
                            task_type, data = q.get_nowait()

                            if task_type == TaskType.INIT:
                                self._init()
                            elif task_type == TaskType.TERMINATE:
                                self._terminate()
                                self.terminate = True
                                break
                            elif task_type == TaskType.POLICY_STEP:
                                self._store_policy_step_request(data)
                            elif task_type == TaskType.UPDATE_WEIGHTS:
                                self._update_weights(data, timing)

                            q.task_done()

                        except Empty:
                            break

                self._handle_policy_steps(timing)

        log.info('Gpu worker timing: %s', timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.join()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        self.process.join(timeout=5)
