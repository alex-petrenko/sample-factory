import sys

from algorithms.utils.arguments import default_cfg
from envs.doom.doom_gym import VizdoomEnv
from envs.doom.doom_utils import doom_env_by_name, make_doom_env_impl


def main():
    env_name = 'doom_dwango5_bots_exploration'
    cfg = default_cfg(env=env_name)
    env = make_doom_env_impl(doom_env_by_name(env_name), cfg=cfg, custom_resolution='1280x720')
    return VizdoomEnv.play_human_mode(env, skip_frames=2)


if __name__ == '__main__':
    sys.exit(main())
