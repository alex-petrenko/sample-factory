import sys

from algorithms.utils.arguments import default_cfg
from envs.create_env import create_env
from utils.utils import log


def main():
    env_name = 'doom_battle'
    env = create_env(env_name, cfg=default_cfg(env=env_name))

    env.reset()
    done = False
    while not done:
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())

    log.info('Done!')


if __name__ == '__main__':
    sys.exit(main())
