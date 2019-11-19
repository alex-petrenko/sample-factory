import math
import threading
from collections import deque
from enum import Enum
from multiprocessing import JoinableQueue, Process
from queue import Queue, Empty

from algorithms.utils.multi_env import safe_get, empty_queue
from utils.timing import Timing
from utils.utils import log, AttrDict


class TaskType(Enum):
    INIT, TERMINATE, RESET, STEP = range(4)


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

    def __init__(self, make_env_func, vector_size=2, num_splits=2, worker_idx=0, use_multiprocessing=False):
        self.make_env_func = make_env_func
        self.is_multiagent = False

        self.vector_size = vector_size
        self.num_splits = num_splits
        assert self.vector_size >= self.num_splits

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
        last_steps = deque([], maxlen=5)
        initialized = False

        while True:
            with timing.add_time('total_time'):
                with timing.add_time('waiting'):
                    msg_type, split_idx, data = safe_get(self.task_queue, timeout=10.0)  #TODO!!! smaller timeout

                if msg_type == TaskType.INIT:
                    self._init(env_vectors)
                    self.task_queue.task_done()
                    continue

                if msg_type == TaskType.TERMINATE:
                    self._terminate(env_vectors)
                    self.task_queue.task_done()
                    break

                # handling actual workload
                envs = env_vectors[split_idx]

                if msg_type == TaskType.RESET:
                    with timing.add_time('reset'):
                        results = [env.reset() for env in envs]  # TODO: put on Plasma storage?
                else:
                    if not initialized:
                        initialized = True
                        # reset the timer
                        timing.total_time = 0.0
                        timing.waiting = 0.0

                    with timing.add_time('work'):
                        # env step
                        actions = data

                        with timing.timeit('one_step'):
                            results = [env.step(action) for env, action in zip(envs, actions)]

                            for i, result in enumerate(results):
                                obs, reward, done, info = result

                                # automatically reset envs upon episode termination
                                is_done = all(done) if self.is_multiagent else done
                                if is_done:
                                    # info will be from the last step of the previous episode, which might not be desirable
                                    obs = envs[i].reset()

                                results[i] = (obs, reward, done, info)

                        # log.info('Submitting result for split %d', split_idx)
                        last_steps.append(timing.one_step)

                self.result_queue.put((split_idx, results))
                self.task_queue.task_done()

        if self.worker_idx <= 1:
            log.info('Env runner %d: timing %s, avg step: %.4f', self.worker_idx, timing, sum(last_steps) / len(last_steps))

    def await_task(self, task_type, split_idx, data=None):
        """Submit a task and block until it's completed."""

        self.task_queue.put((task_type, split_idx, data))
        self.task_queue.join()

        results = safe_get(self.result_queue)
        self.result_queue.task_done()

        return results

    def init(self):
        self.task_queue.put((TaskType.INIT, None, None))
        self.task_queue.join()
        log.info('Env runner %d initialzed...', self.worker_idx)

    def reset(self):
        results = []
        for split in range(self.num_splits):
            _, result = self.await_task(TaskType.RESET, split)
            results.append(result)
        return results

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None, None))
        self.task_queue.join()
        empty_queue(self.result_queue)

        self.process.join(timeout=2.0)
        if self.process.is_alive():
            pass

        return None
