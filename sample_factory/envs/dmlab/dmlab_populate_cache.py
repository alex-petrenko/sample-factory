import signal
import sys
import time

import psutil

from sample_factory.algorithms.dummy_sampler import DummySampler
from sample_factory.algorithms.utils.arguments import maybe_load_from_checkpoint, parse_args
from sample_factory.envs.create_env import create_env
from sample_factory.envs.dmlab import dmlab_level_cache
from sample_factory.envs.dmlab.dmlab30 import DMLAB30_APPROX_NUM_EPISODES_PER_BILLION_FRAMES, DMLAB30_LEVELS_THAT_USE_LEVEL_CACHE
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, AttrDict, get_free_disk_space_mb


DESIRED_TRAINING_LENGTH = int(15e9)


class DmlabLevelGenerator(DummySampler):
    def __init__(self, cfg):
        super().__init__(cfg)

    def sample(self, proc_idx):
        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        timing = Timing()

        psutil.Process().nice(10)

        num_envs = len(DMLAB30_LEVELS_THAT_USE_LEVEL_CACHE)
        assert self.cfg.num_workers % num_envs == 0, f'should have an integer number of workers per env, e.g. {1 * num_envs}, {2 * num_envs}, etc...'
        assert self.cfg.num_envs_per_worker == 1, 'use populate_cache with 1 env per worker'

        with timing.timeit('env_init'):
            env_key = 'env'
            env_desired_num_levels = 0

            global_env_id = proc_idx * self.cfg.num_envs_per_worker
            env_config = AttrDict(worker_index=proc_idx, vector_index=0, env_id=global_env_id)
            env = create_env(self.cfg.env, cfg=self.cfg, env_config=env_config)
            env.seed(global_env_id)

            # this is to track the performance for individual DMLab levels
            if hasattr(env.unwrapped, 'level_name'):
                env_key = env.unwrapped.level_name
                env_level = env.unwrapped.level

                approx_num_episodes_per_1b_frames = DMLAB30_APPROX_NUM_EPISODES_PER_BILLION_FRAMES[env_key]
                num_billions = DESIRED_TRAINING_LENGTH / int(1e9)
                num_workers_for_env = self.cfg.num_workers // num_envs
                env_desired_num_levels = int((approx_num_episodes_per_1b_frames * num_billions) / num_workers_for_env)

                env_num_levels_generated = len(dmlab_level_cache.DMLAB_GLOBAL_LEVEL_CACHE[0].all_seeds[env_level]) // num_workers_for_env

                log.warning('Worker %d (env %s) generated %d/%d levels!', proc_idx, env_key, env_num_levels_generated, env_desired_num_levels)
                time.sleep(4)

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
                        env_num_levels_generated += 1
                        log.debug('Env %s done %d/%d resets', env_key, env_num_levels_generated, env_desired_num_levels)

                    if env_num_levels_generated >= env_desired_num_levels:
                        log.debug('%s finished %d/%d resets, sleeping...', env_key, env_num_levels_generated, env_desired_num_levels)
                        time.sleep(30)  # free up CPU time for other envs

                    # if env does not use level cache, there is no need to run it
                    # let other workers proceed
                    if not env_uses_level_cache:
                        log.debug('Env %s does not require cache, sleeping...', env_key)
                        time.sleep(200)

                    with timing.add_time('report'):
                        now = time.time()
                        if now - last_report > self.report_every_sec:
                            last_report = now
                            frames_since_last_report = total_env_frames - last_report_frames
                            last_report_frames = total_env_frames
                            self.report_queue.put(dict(proc_idx=proc_idx, env_frames=frames_since_last_report))

                            if get_free_disk_space_mb(self.cfg) < 3 * 1024:
                                log.error('Not enough disk space! %d', get_free_disk_space_mb(self.cfg))
                                time.sleep(200)
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


# --algo=DUMMY_SAMPLER --env=dmlab_level_cache --env_frameskip=4 --num_workers=30 --num_envs_per_worker=1 --sample_env_frames=40000000000 --sample_env_frames_per_worker=40000000000 --set_workers_cpu_affinity=False --dmlab_use_level_cache=True --dmlab_renderer=software --experiment=dmlab_populate_cache
