import sys

from sample_factory.algorithms.utils.arguments import default_cfg
from sample_factory.envs.doom.doom_gym import VizdoomEnv
from sample_factory.envs.doom.doom_utils import doom_env_by_name, make_doom_env_impl


def main():
    env_name = 'doom_health_gathering'
    cfg = default_cfg(env=env_name)
    env = make_doom_env_impl(doom_env_by_name(env_name), cfg=cfg, custom_resolution='1280x720')

    # cfg.num_agents = 1
    # cfg.num_bots = 7
    # # cfg.num_humans = 1
    # env = make_doom_multiplayer_env(doom_env_by_name(env_name), cfg=cfg, custom_resolution='1280x720')

    return VizdoomEnv.play_human_mode(env, skip_frames=2, num_actions=15)


if __name__ == '__main__':
    sys.exit(main())
