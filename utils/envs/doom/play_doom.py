import sys

from utils.envs.doom.doom_utils import make_doom_env, doom_env_by_name


def main():
    env = make_doom_env(doom_env_by_name('doom_battle'), mode='test', show_automap=True)
    return env.unwrapped.play_human_mode()


if __name__ == '__main__':
    sys.exit(main())
