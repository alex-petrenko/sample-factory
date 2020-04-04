import time

import psutil
import sys

from algorithms.dummy_sampler.sampler import DummySampler
from algorithms.utils.arguments import maybe_load_from_checkpoint, parse_args
from envs.create_env import create_env
from envs.dmlab.dmlab30 import RANDOM_POLICY_EPISODE_LEN
from utils.timing import Timing
from utils.utils import log, AttrDict, get_free_disk_space_mb

DESIRED_TRAINING_LENGTH = int(1e10)


class DmlabLevelGenerator(DummySampler):
    def __init__(self, cfg):
        super().__init__(cfg)

    def sample(self, proc_idx):
        timing = Timing()

        psutil.Process().nice(10)

        assert self.cfg.num_envs_per_worker == 1, 'use populate_cache with 1 env per worker'

        with timing.timeit('env_init'):
            env_key = 'env'
            env_desired_resets = 0
            env_num_resets = 0

            global_env_id = proc_idx * self.cfg.num_envs_per_worker
            env_config = AttrDict(worker_index=proc_idx, vector_index=0, env_id=global_env_id)
            env = create_env(self.cfg.env, cfg=self.cfg, env_config=env_config)
            env.seed(global_env_id)

            # this is to track the performance for individual DMLab levels
            if hasattr(env.unwrapped, 'level_name'):
                env_key = env.unwrapped.level_name
                env_desired_resets = DESIRED_TRAINING_LENGTH / (RANDOM_POLICY_EPISODE_LEN[env_key] * 30)

            env.reset()
            env_uses_level_cache = env.unwrapped.env_uses_level_cache

            self.report_queue.put(dict(proc_idx=proc_idx, finished_reset=True))

        self.start_event.wait()

        try:
            with timing.timeit('work'):
                last_report = last_report_frames = total_env_frames = 0
                while not self.terminate.value and total_env_frames < self.cfg.sample_env_frames_per_worker:
                    action = env.action_space.sample()
                    with timing.add_time(f'{env_key}.step'):
                        env.step(action)

                    total_env_frames += 1

                    with timing.add_time(f'{env_key}.reset'):
                        env.reset()
                        env_num_resets += 1
                        log.debug('Env %s done %d/%d resets', env_key, env_num_resets, env_desired_resets)

                    if env_num_resets >= env_desired_resets:
                        log.debug('%s finished %d/%d resets, sleeping...', env_key, env_num_resets, env_desired_resets)
                        time.sleep(30)  # free up CPU time for other envs

                    # if env does not use level cache, there is no need to run it
                    # let other workers proceed
                    if not env_uses_level_cache:
                        log.debug('Env %s does not require cache, sleeping...', env_key)
                        time.sleep(300)

                    with timing.add_time('report'):
                        now = time.time()
                        if now - last_report > self.report_every_sec:
                            last_report = now
                            frames_since_last_report = total_env_frames - last_report_frames
                            last_report_frames = total_env_frames
                            self.report_queue.put(dict(proc_idx=proc_idx, env_frames=frames_since_last_report))

                            if get_free_disk_space_mb() < 3 * 1024:
                                log.error('Not enough disk space! %d', get_free_disk_space_mb())
                                time.sleep(300)
        except:
            log.exception('Unknown exception')
            log.error('Unknown exception in worker %d, terminating...', proc_idx)
            self.report_queue.put(dict(proc_idx=proc_idx, crash=True))

        time.sleep(proc_idx * 0.1 + 0.1)
        log.info('Process %d finished sampling. Timing: %s', proc_idx, timing)

        env.close()


def run(cfg):
    cfg = maybe_load_from_checkpoint(cfg)

    algo = DmlabLevelGenerator(cfg)
    algo.initialize()
    status = algo.run()
    algo.finalize()

    log.info('Exit...')
    return status


def main():
    """Script entry point."""
    cfg = parse_args()
    return run(cfg)


if __name__ == '__main__':
    sys.exit(main())


# --algo=DUMMY_SAMPLER --env=dmlab_30 --env_frameskip=4 --num_workers=30 --num_envs_per_worker=1 --sample_env_frames=40000000000 --sample_env_frames_per_worker=40000000000 --set_workers_cpu_affinity=False --dmlab_use_level_cache=True --dmlab_renderer=software --experiment=dmlab_populate_cache
