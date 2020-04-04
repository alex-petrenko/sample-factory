"""
This is a 'fake' RL algorithm that does not do learning of any kind.
All it does is it samples environments as quickly as possible using all CPU cores and random actions.
Designed to establish the upper bound on the RL algorithm throughput.

"""

import ctypes
import multiprocessing
import psutil
import time
from _queue import Empty
from collections import deque
from multiprocessing.sharedctypes import RawValue

import numpy as np

from algorithms.algorithm import AlgorithmBase
from envs.create_env import create_env
from fast_queue import Queue
from utils.timing import Timing
from utils.utils import log, AttrDict, set_process_cpu_affinity, str2bool


class DummySampler(AlgorithmBase):
    @classmethod
    def add_cli_args(cls, parser):
        p = parser

        p.add_argument('--num_workers', default=multiprocessing.cpu_count(), type=int, help='Number of processes to use to sample the environment.')
        p.add_argument('--num_envs_per_worker', default=1, type=int, help='Number of envs on a single CPU sampled sequentially.')

        p.add_argument('--sample_env_frames', default=int(2e6), type=int, help='Stop after sampling this many env frames (this takes frameskip into account)')
        p.add_argument('--sample_env_frames_per_worker', default=int(1e5), type=int, help='Stop after sampling this many env frames per worker (this takes frameskip into account)')

        p.add_argument(
            '--set_workers_cpu_affinity', default=True, type=str2bool,
            help=(
                'Whether to assign workers to specific CPU cores or not. The logic is beneficial for most workloads because prevents a lot of context switching.'
                'However for some environments it can be better to disable it, to allow one worker to use all cores some of the time. This can be the case for some DMLab environments with very expensive episode reset'
                'that can use parallel CPU cores for level generation.'),
        )

    def __init__(self, cfg):
        super().__init__(cfg)

        self.processes = []
        self.terminate = RawValue(ctypes.c_bool, False)

        self.start_event = multiprocessing.Event()
        self.start_event.clear()

        self.report_queue = Queue()
        self.report_every_sec = 1.0
        self.last_report = 0

        self.avg_stats_intervals = (1, 10, 60, 180)
        self.fps_stats = deque([], maxlen=max(self.avg_stats_intervals))

    def initialize(self):
        # creating an environment in the main process tends to fix some very weird issues further down the line
        # https://stackoverflow.com/questions/60963839/importing-opencv-after-importing-pytorch-messes-with-cpu-affinity
        # do not delete this unless you know what you're doing
        tmp_env = create_env(self.cfg.env, cfg=self.cfg, env_config=None)
        tmp_env.close()

        for i in range(self.cfg.num_workers):
            p = multiprocessing.Process(target=self.sample, args=(i, ))
            self.processes.append(p)

    def sample(self, proc_idx):
        timing = Timing()

        if self.cfg.set_workers_cpu_affinity:
            set_process_cpu_affinity(proc_idx, self.cfg.num_workers)
        initial_cpu_affinity = psutil.Process().cpu_affinity()
        psutil.Process().nice(10)

        with timing.timeit('env_init'):
            envs = []
            env_key = ['env' for _ in range(self.cfg.num_envs_per_worker)]

            for env_idx in range(self.cfg.num_envs_per_worker):
                global_env_id = proc_idx * self.cfg.num_envs_per_worker + env_idx
                env_config = AttrDict(worker_index=proc_idx, vector_index=env_idx, env_id=global_env_id)
                env = create_env(self.cfg.env, cfg=self.cfg, env_config=env_config)
                env.seed(global_env_id)
                envs.append(env)

                # this is to track the performance for individual DMLab levels
                if hasattr(env.unwrapped, 'level_name'):
                    env_key[env_idx] = env.unwrapped.level_name

            episode_length = [0 for _ in envs]
            episode_lengths = [deque([], maxlen=20) for _ in envs]

        try:
            with timing.timeit('first_reset'):
                for env_idx, env in enumerate(envs):
                    env.reset()
                    log.info('Process %d finished resetting %d/%d envs', proc_idx, env_idx + 1, len(envs))

                self.report_queue.put(dict(proc_idx=proc_idx, finished_reset=True))

            self.start_event.wait()

            with timing.timeit('work'):
                last_report = last_report_frames = total_env_frames = 0
                while not self.terminate.value and total_env_frames < self.cfg.sample_env_frames_per_worker:
                    for env_idx, env in enumerate(envs):
                        action = env.action_space.sample()
                        with timing.add_time(f'{env_key[env_idx]}.step'):
                            obs, reward, done, info = env.step(action)

                        num_frames = info.get('num_frames', 1)
                        total_env_frames += num_frames
                        episode_length[env_idx] += num_frames

                        if done:
                            with timing.add_time(f'{env_key[env_idx]}.reset'):
                                env.reset()

                            episode_lengths[env_idx].append(episode_length[env_idx])
                            episode_length[env_idx] = 0

                    with timing.add_time('report'):
                        now = time.time()
                        if now - last_report > self.report_every_sec:
                            last_report = now
                            frames_since_last_report = total_env_frames - last_report_frames
                            last_report_frames = total_env_frames
                            self.report_queue.put(dict(proc_idx=proc_idx, env_frames=frames_since_last_report))

            # Extra check to make sure cpu affinity is preserved throughout the execution.
            # I observed weird effect when some environments tried to alter affinity of the current process, leading
            # to decreased performance.
            # This can be caused by some interactions between deep learning libs, OpenCV, MKL, OpenMP, etc.
            # At least user should know about it if this is happening.
            cpu_affinity = psutil.Process().cpu_affinity()
            assert initial_cpu_affinity == cpu_affinity, \
                f'Worker CPU affinity was changed from {initial_cpu_affinity} to {cpu_affinity}!' \
                f'This can significantly affect performance!'

        except:
            log.exception('Unknown exception')
            log.error('Unknown exception in worker %d, terminating...', proc_idx)
            self.report_queue.put(dict(proc_idx=proc_idx, crash=True))

        time.sleep(proc_idx * 0.1 + 0.1)
        log.info('Process %d finished sampling. Timing: %s', proc_idx, timing)

        for env_idx, env in enumerate(envs):
            log.warning('Level %s avg episode len %d', env_key[env_idx], np.mean(episode_lengths[env_idx]))

        for env in envs:
            env.close()

    def report(self, env_frames):
        now = time.time()
        self.last_report = now

        self.fps_stats.append((now, env_frames))
        if len(self.fps_stats) <= 1:
            return

        fps = []
        for avg_interval in self.avg_stats_intervals:
            past_moment, past_frames = self.fps_stats[max(0, len(self.fps_stats) - 1 - avg_interval)]
            fps.append((env_frames - past_frames) / (now - past_moment))

        fps_str = []
        for interval, fps_value in zip(self.avg_stats_intervals, fps):
            fps_str.append(f'{int(interval * self.report_every_sec)} sec: {fps_value:.1f}')
        fps_str = f'({", ".join(fps_str)})'
        log.info('Sampling FPS: %s. Total frames collected: %d', fps_str, env_frames)

    def run(self):
        for p in self.processes:
            p.start()

        finished_reset = np.zeros([self.cfg.num_workers], dtype=np.bool)
        while not all(finished_reset):
            try:
                msg = self.report_queue.get(timeout=0.1)
                if 'finished_reset' in msg:
                    finished_reset[msg['proc_idx']] = True
            except Empty:
                pass
        log.debug('All workers finished reset!')

        self.start_event.set()

        start = time.time()
        env_frames = 0

        while not self.terminate.value:
            try:
                msgs = self.report_queue.get_many(timeout=0.1)
                for msg in msgs:
                    if 'crash' in msg:
                        self.terminate.value = True
                        log.error('Terminating due to process %d crashing...', msg['proc_idx'])
                        break

                    env_frames += msg['env_frames']

                if env_frames >= self.cfg.sample_env_frames:
                    self.terminate.value = True
            except Empty:
                pass
            except KeyboardInterrupt:
                self.terminate.value = True

            if time.time() - self.last_report > self.report_every_sec:
                self.report(env_frames)

            all_dead = True
            for p in self.processes:
                if p.is_alive():
                    all_dead = False

            if all_dead:
                self.terminate.value = True

        total_time = time.time() - start
        log.info('Collected %d frames in %.1f s, avg FPS: %.1f', env_frames, total_time, env_frames / total_time)
        log.debug('Done sampling...')

    def finalize(self):
        try:
            self.report_queue.get_many_nowait()
        except Empty:
            pass

        log.debug('Joining worker processes...')
        for p in self.processes:
            p.join()
        log.debug('Done joining!')


# --algo=DUMMY_SAMPLER --env=doom_benchmark --env_frameskip=4 --res_w=128 --res_h=72 --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=10 --sample_env_frames=2000000 --experiment=test
# --algo=DUMMY_SAMPLER --env=dmlab_benchmark --env_frameskip=4 --num_workers=20 --num_envs_per_worker=10 --sample_env_frames=2000000 --experiment=test
