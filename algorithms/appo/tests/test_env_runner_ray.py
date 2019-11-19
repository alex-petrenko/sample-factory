"""
Ray does not work with unittests, so this is a standalone testing script.

"""
import pickle

import ray.pyarrow_files.pyarrow as pa

import sys
import time
from collections import deque

import ray
from ray.pyarrow_files.pyarrow import plasma

from algorithms.appo.env_runner import VectorEnvRunner, TaskType
from algorithms.utils.algo_utils import num_env_steps
from algorithms.utils.multi_env import safe_get
from envs.doom.doom_utils import make_doom_env
from envs.tests.test_envs import default_doom_cfg
from utils.timing import Timing
from utils.utils import log, AttrDict


"""
Benchmark (num_splits=2, unless specified):

With vector=10
8 workers: 19771 FPS
16 workers: 20971 FPS
20 workers: 20002 FPS
32 workers: 19623 FPS

With vector=20
20 workers: 24005 FPS

With vector=30
20 workers: 29533 FPS (no obs: 31231 FPS)
20 workers, up to 8 tasks in ray.wait(): 34331 FPS (??!!)
                                         34575 FPS
32 workers: 33387 FPS

20 workers split=1: 32126 FPS (whaaat?) Fewer synchronizations = better performance
20 workers split=3: 27135 FPS
"""


@ray.remote(num_cpus=0.3)
class Actor:
    def __init__(self, make_env_func, vector_size, num_splits, worker_id, plasma_name):
        self.worker_id = worker_id
        self.num_frames = 0
        self.make_env_func = make_env_func
        self.vector_size = vector_size
        self.num_splits = num_splits

        self.action_space = None
        self.env_runner = None

        self.timing = Timing()
        self.got_result = time.time()
        self.action_times = deque([], maxlen=10)

        self.plasma_client = plasma.connect(plasma_name)

    def ensure_initialized(self):
        return self.worker_id

    def reset(self):
        tmp_env = self.make_env_func(None)
        self.action_space = tmp_env.action_space
        tmp_env.close()
        del tmp_env

        self.env_runner = VectorEnvRunner(
            self.make_env_func, vector_size=self.vector_size, num_splits=self.num_splits, worker_idx=self.worker_id,
        )
        self.env_runner.init()

        # initiate the rollout
        obs = self.env_runner.reset()
        log.info('Reset worker %d', self.worker_id)

        return {
            'worker': self.worker_id,
            'splits': list(range(len(obs))),
            'obs': obs,
        }

    def advance_rollout(self, split, actions, wait=True):
        # submitting actions for a split when they are ready
        # log.info('Advance rollout for split %d (worker %d)', split, self.worker_id)
        self.env_runner.task_queue.put((TaskType.STEP, split, actions))
        self.action_times.append(time.time() - self.got_result)

        if not wait:
            return None

        results = safe_get(self.env_runner.result_queue)
        self.got_result = time.time()
        new_split, results = results
        # log.info('Got results for split %d (worker %d)', new_split, self.worker_id)

        self.env_runner.result_queue.task_done()

        # log.info('Calculating actions for split %d', split)

        with self.timing.time_avg('ray_put_obs'):
            # obs_obj = ray.put([r[0] for r in results])
            obs_obj = [r[0] for r in results]
            # obs_obj = self.plasma_client.put(obs_obj, None, memcopy_threads=1)

        return {
            'worker': self.worker_id,
            'split': new_split,
            'obs': obs_obj,
            'infos': [r[-1] for r in results],
        }

    def terminate(self):
        if self.worker_id <= 1:
            log.info('Avg action time %.4f', sum(self.action_times) / len(self.action_times))
            log.info('Actor timing: %s', self.timing)
        self.env_runner.close()


def make_env_singleplayer(env_config):
    return make_doom_env('doom_battle_hybrid', cfg=default_doom_cfg(), env_config=env_config)


def make_env_bots(env_config):
    log.info('Create host env with cfg: %r', env_config)
    return make_doom_env('doom_dwango5_bots_experimental', cfg=default_doom_cfg(), env_config=env_config)


def env_runner_multi():
    ray.init(local_mode=False)
    worker = ray.worker.global_worker
    plasma_socket_name = worker.node.plasma_store_socket_name
    plasma_client = plasma.connect(plasma_socket_name)
    serialization_context = pa.default_serialization_context()

    tmp_env = make_env_singleplayer(None)
    action_space = tmp_env.action_space

    def generate_actions(obs_, infos_, split_):
        # log.info('Generate actions for split %d', split_)
        num_obs = len(infos_)
        actions_ = [action_space.sample()] * num_obs
        return actions_

    tmp_env.close()

    num_workers = 20
    vector_size = 20
    num_splits = 2
    workers = []
    for i in range(num_workers):
        worker = Actor.remote(make_env_singleplayer, vector_size, num_splits, i, plasma_socket_name)
        ray.get(worker.ensure_initialized.remote())
        workers.append(worker)

    max_parallel_init = 10
    reset_results = []
    for i in range(0, num_workers, max_parallel_init):
        reset_tasks = [
            w.reset.remote() for w in workers[i:i + max_parallel_init]
        ]
        reset_results.extend([ray.get(t) for t in reset_tasks])

    tasks = []
    for res in reset_results:
        res = AttrDict(res)
        worker, splits, obs = res.worker, res.splits, res.obs

        for obs_split, split in zip(obs, splits):
            actions = generate_actions(obs_split, [{}] * len(obs_split), split)
            wait = split >= len(splits) - 1
            tasks.append(workers[worker].advance_rollout.remote(split, actions, wait))

    num_frames = 0
    timing = Timing()
    with timing.timeit('experience'):
        while num_frames < 200000:
            finished, tasks = ray.wait(tasks, num_returns=min(len(tasks), 8), timeout=0.001)

            for task in finished:
                res = ray.get(task)
                if res is None:
                    continue

                res = AttrDict(res)
                worker, split, obs, infos = res.worker, res.split, res.obs, res.infos
                actions = generate_actions(obs, infos, split)

                # with timing.add_time('pickle'):
                #     pickled = pickle.dumps(obs)
                #     pass
                #
                # with timing.add_time('pa_serialize'):
                #     obs_serialized = pa.serialize(obs, serialization_context)
                #     pass
                #
                with timing.time_avg('ray_put', average=500):
                    object_id = plasma_client.put(obs, None, serialization_context=serialization_context)
                #
                # with timing.add_time('ray_put_empty'):
                #     empty_obj_id = plasma_client.put([0], None, serialization_context=serialization_context)
                #
                # with timing.add_time('ray_get'):
                #     result = plasma_client.get(object_id, -1, serialization_context=serialization_context)
                #     pass

                # this will usually be delayed
                tasks.append(workers[worker].advance_rollout.remote(split, actions, True))

                num_frames += num_env_steps(infos)

    terminate_tasks = [w.terminate.remote() for w in workers]
    done, _ = ray.wait(terminate_tasks, num_returns=len(terminate_tasks))
    log.info('Terminated %d workers...', len(terminate_tasks))

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
