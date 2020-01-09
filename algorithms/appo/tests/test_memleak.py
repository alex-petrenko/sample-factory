import sys

from algorithms.appo.actor_worker import make_env_func
from envs.tests.test_envs import default_doom_cfg
from utils.utils import AttrDict, memory_consumption_mb, log


def main():
    cfg = default_doom_cfg()
    cfg.env = 'doom_ssl2_duel'
    env_config = AttrDict({'worker_index': 0, 'vector_index': 0})
    env = make_env_func(cfg, env_config)

    for i in range(int(1e6)):
        action = [env.action_space.sample()] * env.num_agents
        obs, rew, done, info = env.step(action)
        if all(done):
            env.reset()

        if i % 1000 == 0:
            mem = memory_consumption_mb()
            log.debug('Memory used: %.3f (%d)', mem, i)


if __name__ == '__main__':
    sys.exit(main())
