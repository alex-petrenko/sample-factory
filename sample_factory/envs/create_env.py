from __future__ import annotations

from typing import Optional

from sample_factory.algo.utils.context import global_env_registry
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log


# TODO: type hint for the return type?
def create_env(
    full_env_name: str,
    cfg: Optional[Config] = None,
    env_config: Optional[AttrDict] = None,
):
    """
    Factory function that creates environment instances.
    :param full_env_name: complete name of the environment
    :param cfg: namespace with full system configuration, output of argparser (or AttrDict when loaded from JSON)
    :param env_config: AttrDict with additional system information:
    env_config = AttrDict(worker_index=self.worker_idx, vector_index=vector_idx, env_id=env_id)

    :return: environment instance
    """

    env_registry = global_env_registry()

    if full_env_name not in env_registry:
        msg = f"Env name {full_env_name} is not registered. See register_env()!"
        log.error(msg)
        log.debug(f"Registered env names: {env_registry.keys()}")
        raise ValueError(msg)

    env_factory = env_registry[full_env_name]
    env = env_factory(full_env_name, cfg, env_config)
    return env
