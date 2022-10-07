import sys

from sample_factory.algo.utils.rl_utils import make_dones
from sample_factory.cfg.arguments import default_cfg
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log


def main():
    env_name = "doom_battle"
    env = create_env(env_name, cfg=default_cfg(env=env_name))

    env.reset()
    done = False
    while not done:
        env.render()
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        done = make_dones(terminated, truncated)

    log.info("Done!")


if __name__ == "__main__":
    sys.exit(main())
