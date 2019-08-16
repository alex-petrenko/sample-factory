import sys

from algorithms.utils.multi_env import MultiEnv
from envs.dmlab.dmlab_utils import DmlabGymEnv
from utils.utils import log


def main():
    def make_env(env_config):
        env = DmlabGymEnv('contributed/dmlab30/rooms_watermaze', 4)
        return env

    num_envs = 64
    num_workers = 16
    multi_env = MultiEnv(num_envs, num_workers, make_env, stats_episodes=100)
    num_resets = 0

    try:
        while True:
            multi_env.reset()
            num_resets += 1
            num_envs_generated = num_resets * num_envs
            log.info('Generated %d environments...', num_envs_generated)
    except (Exception, KeyboardInterrupt, SystemExit):
        log.exception('Interrupt...')
    finally:
        log.info('Closing env...')
        multi_env.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
