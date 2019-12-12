import time
from queue import Empty, Queue
from threading import Thread

import numpy as np
import ray.pyarrow_files.pyarrow as pa
import torch
from ray.pyarrow_files.pyarrow import plasma
from torch.multiprocessing import Process

from algorithms.appo.appo_utils import TaskType, dict_of_lists_append
from algorithms.appo.model import ActorCritic
from algorithms.utils.multi_env import safe_get
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

        self.num_requests = 0
        self.terminate = False

        self.latest_policy_version = 0

        self.process = Process(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _should_log(self):
        log_rate = 50
        return self.num_requests % log_rate == 0

    def _init(self):
        log.info('GPU worker %d initialized', self.worker_idx)

    def _terminate(self):
        # log.info('GPU worker %d terminated', self.worker_idx)
        pass

    def _handle_policy_step(self, requests, timing):
        # log.info('Num pending requests: %d', len(requests))
        if len(requests) <= 0:
            return

        self.num_requests += 1
        with timing.add_time('policy_step'):
            with timing.add_time('deserialize'):
                observations = AttrDict()
                rnn_states = []
                num_obs_per_actor = []

                for request in requests:
                    request = self.plasma_client.get(
                        request, -1, serialization_context=self.serialization_context,
                    )

                    actor_idx = request['worker_idx']
                    split_idx = request['split_idx']
                    num_inputs, policy_input = request['policy_inputs']

                    dict_of_lists_append(observations, policy_input['obs'])
                    rnn_states.append(policy_input['rnn_states'])
                    num_obs_per_actor.append((actor_idx, split_idx, num_inputs))

            with torch.no_grad():
                with timing.add_time('to_device'):
                    for key, x in observations.items():
                        observations[key] = torch.from_numpy(np.concatenate(x)).to(self.device).float()

                    rnn_states = np.concatenate(rnn_states)
                    rnn_states = torch.from_numpy(rnn_states).to(self.device).float()

                # if self._should_log():
                #     log.info(
                #         'Forward pass for policy %d, num observations in a batch %d, GPU worker %d',
                #         policy_id, rnn_states.shape[0], self.worker_idx,
                #     )

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

        pause = False
        pending_requests = []

        while not self.terminate:
            weight_update = None
            work_done = False

            while True:
                try:
                    task_type, data = self.task_queue.get_nowait()
                    if task_type == TaskType.INIT:
                        self._init()
                    elif task_type == TaskType.TERMINATE:
                        self._terminate()
                        self.terminate = True
                        break
                    elif task_type == TaskType.POLICY_STEP:
                        pending_requests.append(data)

                    self.task_queue.task_done()
                    work_done = True

                except Empty:
                    break

            if not pause:
                self._handle_policy_step(pending_requests, timing)
                pending_requests = []

            while True:
                try:
                    task_type, data = self.weight_queue.get_nowait()
                    if task_type == TaskType.UPDATE_WEIGHTS:
                        weight_update = data
                    elif task_type == TaskType.TOO_MUCH_DATA:
                        pause = data
                        log.debug('Pause: %r', pause)

                    self.weight_queue.task_done()
                    work_done = True
                except Empty:
                    break

            self._update_weights(weight_update, timing)

            # TODO: wait on both queues (use select.select)
            if not work_done:
                with timing.add_time('gpu_waiting'):
                    time.sleep(0.001)

        log.info('Gpu worker timing: %s', timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.join()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        self.process.join(timeout=5)
