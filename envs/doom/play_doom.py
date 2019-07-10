import sys

from envs.doom.doom_gym import VizdoomEnv
from envs.doom.doom_utils import doom_env_by_name, DEFAULT_FRAMESKIP, make_doom_multiagent_env


def main():
    env = make_doom_multiagent_env(doom_env_by_name('doom_dwango5_bots'), env_config=None, mode='human')
    return VizdoomEnv.play_human_mode(env, skip_frames=DEFAULT_FRAMESKIP)


if __name__ == '__main__':
    sys.exit(main())
