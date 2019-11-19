"""
Ray does not work with unittests, so this is a standalone testing script.

"""
import math
import select
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
from enum import Enum
from multiprocessing import Process, JoinableQueue
from queue import Queue

import ray
import ray.pyarrow_files.pyarrow as pa
from ray.pyarrow_files.pyarrow import plasma

from algorithms.appo.appo import ActorCritic
from algorithms.utils.algo_utils import num_env_steps
from algorithms.utils.arguments import default_cfg
from algorithms.utils.multi_env import safe_get, empty_queue
from envs.doom.doom_utils import make_doom_env
from envs.tests.test_envs import default_doom_cfg
from utils.timing import Timing
from utils.utils import log, AttrDict




class TaskType(Enum):
    INIT, TERMINATE, RESET, STEP, ACTIONS = range(5)


class VectorEnvRunner:
    """
    Works with an array (vector) of environments that is processes in portions.
    Simple case, env vector is split into two parts:
    1. Do an environment step in the 1st half of the vector (envs 1..N/2)
    2. Send observations to a queue for action generation elsewhere (e.g. on a GPU worker)
    3. Immediately start processing second half of the vector (envs N/2+1..N)
    4. By the time second half is processed, actions for the 1st half should be ready. Immediately start processing
    the 1st half of the vector again.

    As a result, if action generation is fast enough, this env runner should be busy 100% of the time
    calculating env steps, without waiting for actions.
    This is somewhat similar to double-buffered rendering in computer graphics.

    """

    def __init__(self, make_env_func, vector_size=2, num_splits=2, worker_idx=0, plasma_socket_name=None, use_multiprocessing=False):
        self.make_env_func = make_env_func
        self.plasma_socket_name = plasma_socket_name
        self.is_multiagent = False

        self.vector_size = vector_size
        self.num_splits = num_splits
        assert self.vector_size >= self.num_splits

        self.plasma_client = None
        self.serialization_context = None

        self.worker_idx = worker_idx

        if use_multiprocessing:
            self.task_queue, self.result_queue = JoinableQueue(), JoinableQueue()
            self.process = Process(target=self.run, daemon=True)
        else:
            log.debug('Not using multiprocessing!')
            self.task_queue, self.result_queue = Queue(), Queue()
            # shouldn't be named "process" here, but who cares
            self.process = threading.Thread(target=self.run)

        self.process.start()

    def _init(self, env_vectors):
        env_i = 0
        env_vector = []
        num_envs_per_split = math.ceil(self.vector_size / self.num_splits)

        log.info('Initializing envs for env runner %d...', self.worker_idx)
        while env_i < self.vector_size:
            env_config = AttrDict({'worker_index': self.worker_idx, 'vector_index': env_i})
            env = self.make_env_func(env_config)

            if hasattr(env, 'num_agents'):
                self.is_multiagent = True

            env.seed(self.worker_idx * 1000 + env_i)

            env_vector.append(env)
            if len(env_vector) >= num_envs_per_split:
                env_vectors.append(env_vector)
                env_vector = []

            env_i += 1

        if len(env_vector) > 0:
            env_vectors.append(env_vector)

        self.plasma_client = plasma.connect(self.plasma_socket_name)
        self.serialization_context = pa.default_serialization_context()

        log.info('Env runner %d split sizes: %r', self.worker_idx, [len(v) for v in env_vectors])

    @staticmethod
    def _terminate(env_vectors):
        for env_vector in env_vectors:
            for env in env_vector:
                env.close()

    def run(self):
        log.info('Initializing vector env runner %d...', self.worker_idx)

        env_vectors = []
        timing = Timing()
        initialized = False

        while True:
            with timing.add_time('total_time'):
                with timing.add_time('waiting'):
                    timeout = 1 if initialized else 1e3
                    task_type, data = safe_get(self.task_queue, timeout=timeout)

                if task_type == TaskType.INIT:
                    self._init(env_vectors)
                    self.task_queue.task_done()
                    continue

                if task_type == TaskType.TERMINATE:
                    self._terminate(env_vectors)
                    self.task_queue.task_done()
                    break

                # handling actual workload
                if task_type == TaskType.RESET:
                    with timing.add_time('reset'):
                        obs = []
                        for env_vector in env_vectors:
                            obs.append([e.reset() for e in env_vector])

                        result = dict(splits=list(range(self.num_splits)), obs=obs)
                else:
                    if not initialized:
                        initialized = True
                        # reset the timer
                        timing.total_time = 0.0
                        timing.waiting = 0.0

                    split_idx, actions = data
                    envs = env_vectors[split_idx]

                    with timing.add_time('work'):
                        with timing.time_avg('one_step'):
                            env_outputs = [env.step(action) for env, action in zip(envs, actions)]
                            # log.info('Env step completed split %d worker %d', split_idx, self.worker_idx)

                            num_outputs = len(env_outputs)
                            observations = []
                            misc_results = []

                            for i, env_output in enumerate(env_outputs):
                                obs, reward, done, info = env_output

                                # automatically reset envs upon episode termination
                                is_done = all(done) if self.is_multiagent else done
                                if is_done:
                                    # info will be from the last step of the previous episode, which might not be desirable
                                    obs = envs[i].reset()

                                # env_outputs[i] = (obs, reward, done, info)
                                observations.append(obs)
                                misc_results.append((reward, done, info))

                            # env_outputs = self.plasma_client.put(env_outputs, None, serialization_context=self.serialization_context)

                            observations = self.plasma_client.put(observations, None, serialization_context=self.serialization_context)
                            result = dict(split=split_idx, observations=observations, misc_results=misc_results, num_outputs=num_outputs)

                result['worker_idx'] = self.worker_idx
                result['task_type'] = task_type

                self.result_queue.put(result)
                self.task_queue.task_done()

        if self.worker_idx <= 1:
            log.info('Env runner %d: timing %s', self.worker_idx, timing)

    def await_task(self, task_type, split_idx, data=None):
        """Submit a task and block until it's completed."""

        self.task_queue.put((task_type, split_idx, data))
        self.task_queue.join()

        results = safe_get(self.result_queue)
        self.result_queue.task_done()

        return results

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.join()
        log.info('Env runner %d initialzed...', self.worker_idx)

    def reset(self):
        results = []
        for split in range(self.num_splits):
            _, result = self.await_task(TaskType.RESET, split)
            results.append(result)
        return results

    def request_reset(self):
        self.task_queue.put((TaskType.RESET, None))

    def request_step(self, split, actions):
        data = (split, actions)
        self.task_queue.put((TaskType.STEP, data))

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))
        self.task_queue.join()
        empty_queue(self.result_queue)
        self.process.join(timeout=2.0)


class GPUWorker:
    def __init__(self, obs_space, action_space, worker_idx=0, plasma_socket_name=None, use_multiprocessing=True):
        self.free = True
        self.plasma_socket_name = plasma_socket_name

        self.worker_idx = worker_idx
        self.obs_space = obs_space
        self.action_space = action_space

        self.plasma_client = None
        self.serialization_context = None

        self.device = None
        self.actor_critic = None

        if use_multiprocessing:
            self.task_queue, self.result_queue = JoinableQueue(), JoinableQueue()
            self.process = Process(target=self.run, daemon=True)
        else:
            log.debug('Not using multiprocessing!')
            self.task_queue, self.result_queue = Queue(), Queue()
            # shouldn't be named "process" here, but who cares
            self.process = threading.Thread(target=self.run)

        self.process.start()

    def run(self):
        self.plasma_client = plasma.connect(self.plasma_socket_name)
        self.serialization_context = pa.default_serialization_context()

        cfg = default_cfg(algo='APPO')
        self.actor_critic = ActorCritic(self.obs_space, self.action_space, cfg)
        self.device = torch.device('cuda')
        self.actor_critic.to(self.device)

        num_requests = 0

        while True:
            data = safe_get(self.task_queue)
            num_requests += 1
            results = []
            observations = []

            timing = Timing()
            with timing.timeit('forward'):
                for action_request in data:
                    num_obs, split, actor, obs = action_request

                    obs = self.plasma_client.get(obs, -1, serialization_context=self.serialization_context)
                    observations.extend(obs)

                obs_dict = AttrDict()
                if isinstance(observations[0], (dict, OrderedDict)):
                    for key in observations[0].keys():
                        if not isinstance(observations[0][key], str):
                            obs_dict[key] = [o[key] for o in observations]
                else:
                    # handle flat observations also as dict
                    obs_dict.obs = observations

                batch_size = len(obs_dict.obs)
                rnn_states = np.zeros((batch_size, cfg.hidden_size), dtype=np.float32)
                rnn_states = torch.from_numpy(rnn_states).to(self.device).float()

                with torch.no_grad():
                    for key, x in obs_dict.items():
                        obs_dict[key] = torch.from_numpy(np.stack(x)).to(self.device).float()

                    policy_outputs = self.actor_critic(obs_dict, rnn_states)
                    actions = policy_outputs.actions.cpu().numpy()

                curr_idx = 0
                for action_request in data:
                    num_obs, split, actor, obs = action_request

                    actions_actor = actions[curr_idx:curr_idx + num_obs]
                    result = dict(actions=actions_actor, split=split, actor=actor)
                    results.append(result)

            if num_requests % 20 == 1:
                log.info('Actions generated for %d obs in %s', len(observations), timing)

            self.result_queue.put(dict(task_type=TaskType.ACTIONS, results=results))
            self.task_queue.task_done()


def make_env_singleplayer(env_config):
    return make_doom_env('doom_battle_hybrid', cfg=default_doom_cfg(), env_config=env_config)


def make_env_bots(env_config):
    log.info('Create host env with cfg: %r', env_config)
    return make_doom_env('doom_dwango5_bots_experimental', cfg=default_doom_cfg(), env_config=env_config)


def generate_actions(obs_, infos_, action_space):
    num_obs = len(infos_)
    actions_ = [action_space.sample()] * num_obs
    return actions_


def env_runner_multi():
    ray.init(local_mode=False)
    global_worker = ray.worker.global_worker
    plasma_socket_name = global_worker.node.plasma_store_socket_name

    tmp_env = make_env_singleplayer(None)
    obs_space = tmp_env.observation_space
    action_space = tmp_env.action_space
    tmp_env.close()

    log.info('Init GPU workers')
    gpu_workers = []
    num_gpu_workers = 2
    for i in range(num_gpu_workers):
        gpu_workers.append(GPUWorker(obs_space, action_space, i, plasma_socket_name=plasma_socket_name))

    log.info('Init Actors')

    num_workers = 32
    vector_size = 40
    # num_workers = 2
    # vector_size = 4
    num_splits = 2
    workers = []
    for i in range(num_workers):
        worker = VectorEnvRunner(make_env_singleplayer, vector_size, num_splits, i, plasma_socket_name=plasma_socket_name, use_multiprocessing=True)
        workers.append(worker)

    workers_by_handle = dict()
    for w in workers:
        workers_by_handle[w.result_queue._reader._handle] = w
    for w in gpu_workers:
        workers_by_handle[w.result_queue._reader._handle] = w

    max_parallel_init = 10
    reset_results = []
    for i in range(0, num_workers, max_parallel_init):
        for w in workers[i:i + max_parallel_init]:
            w.init()
            w.request_reset()

        for w in workers[i:i + max_parallel_init]:
            reset_result = safe_get(w.result_queue)
            reset_results.append(reset_result)
            log.info('Reset worker %d complete...', w.worker_idx)

    for res in reset_results:
        res = AttrDict(res)
        worker_idx, splits, obs = res.worker_idx, res.splits, res.obs
        for obs_split, split in zip(obs, splits):
            actions = generate_actions(obs_split, [{}] * len(obs_split), action_space)
            workers[worker_idx].request_step(split, actions)

    log.info('Collecting experience...')

    num_frames = 0
    timing = Timing()
    queues = [w.result_queue._reader for w in workers]
    queues.extend([w.result_queue._reader for w in gpu_workers])

    action_requests = []
    with timing.timeit('experience'):
        while num_frames < 200000:
            ready, _, _ = select.select(queues, [], [], 0.001)

            for ready_queue in ready:
                w = workers_by_handle[ready_queue._handle]
                result = safe_get(w.result_queue)
                result = AttrDict(result)
                task_type = result.task_type

                if task_type == TaskType.STEP:
                    split, obs, misc_results = result.split, result.observations, result.misc_results
                    infos = [r[-1] for r in misc_results]
                    actor = result.worker_idx
                    num_outputs = result.num_outputs

                    action_requests.append((num_outputs, split, actor, obs))

                    # log.info('Received result for split %d, worker %d', split, actor)
                    # gpu_workers[0].task_queue.put((num_outputs, split, actor))

                    # actions = generate_actions(obs, infos, action_space)
                    # w.request_step(split, actions)

                    num_frames += num_env_steps(infos)
                elif task_type == TaskType.ACTIONS:
                    for r in result.results:
                        actions, split, actor = r['actions'], r['split'], r['actor']
                        w = workers[actor]
                        w.request_step(split, actions)
                    gpu_workers[0].free = True
                    # log.info('Env step requested for split %d worker %d', split, actor)
                else:
                    log.warning('Unknown task %d', task_type)

            if len(action_requests) > 0 and gpu_workers[0].free:
                gpu_workers[0].task_queue.put(action_requests)
                gpu_workers[0].free = False
                action_requests = []

    for w in workers:
        w.close()

    fps = num_frames / timing.experience
    log.info('Collected %d, FPS: %.1f', num_frames, fps)
    log.info('Timing: %s', timing)

    time.sleep(0.1)
    ray.shutdown()
    log.info('Done!')


def main():
    """Script entry point."""
    return env_runner_multi()


if __name__ == '__main__':
    sys.exit(main())
