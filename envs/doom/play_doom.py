import sys

from envs.doom.doom_utils import make_doom_env, doom_env_by_name, DEFAULT_FRAMESKIP


def main():
    env = make_doom_env(doom_env_by_name('doom_battle'), mode='human', show_automap=True)
    return env.unwrapped.play_human_mode(skip_frames=DEFAULT_FRAMESKIP)


if __name__ == '__main__':
    sys.exit(main())
