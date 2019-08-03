import sys

from envs.doom.doom_gym import VizdoomEnv
from envs.doom.doom_utils import doom_env_by_name, make_doom_env_impl


def main():
    env = make_doom_env_impl(
        doom_env_by_name('doom_two_colors_easy'), env_config=None, custom_resolution='1280x720',
    )
    return VizdoomEnv.play_human_mode(env, skip_frames=4)


if __name__ == '__main__':
    sys.exit(main())
