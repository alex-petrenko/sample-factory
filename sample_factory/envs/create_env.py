from sample_factory.envs.env_registry import global_env_registry


def create_env(full_env_name, cfg=None, env_config=None):
    """
    Factory function that creates environment instances.
    Matches full_env_name with env family prefixes registered in the REGISTRY and calls make_env_func()
    for the first match.

    :param full_env_name: complete name of the environment, starting with the prefix of registered environment family,
    e.g. atari_breakout, or doom_battle. Passed to make_env_func() for further processing by the specific env family
    factory (see doom_utils.py or dmlab_env.py)
    :param cfg: namespace with full system configuration, output of argparser (or AttrDict when loaded from JSON)
    :param env_config: AttrDict with additional system information:
    env_config = AttrDict(worker_index=self.worker_idx, vector_index=vector_idx, env_id=env_id)

    :return: environment instance
    """

    env_registry = global_env_registry()
    env_registry_entry = env_registry.resolve_env_name(full_env_name)
    env = env_registry_entry.make_env_func(full_env_name, cfg=cfg, env_config=env_config)
    return env
