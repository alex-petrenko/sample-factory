from algorithms.utils.multi_env import MultiEnv
from envs.doom.multiplayer.doom_multiagent_wrapper import MultiAgentEnvAggregator
from utils.utils import log


def create_multi_env(num_envs, num_workers, make_env_func, stats_episodes):
    tmp_env = make_env_func(None)
    is_multiagent = hasattr(tmp_env, 'num_agents') and tmp_env.num_agents > 1

    if is_multiagent:
        assert num_envs % tmp_env.num_agents == 0
        log.debug('Num envs %d agents %d', num_envs, tmp_env.num_agents)
        num_envs = num_envs // tmp_env.num_agents
        multi_env = MultiAgentEnvAggregator(num_envs, num_workers, make_env_func, stats_episodes)
    else:
        multi_env = MultiEnv(num_envs, num_workers, make_env_func, stats_episodes)

    tmp_env.close()

    return multi_env


class EnvCriticalError(Exception):
    pass
