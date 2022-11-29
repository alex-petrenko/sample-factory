from typing import Optional

from sample_factory.utils.utils import log

try:
    import envpool
except ImportError as e:
    print(e)
    print("Trying to import envpool when it is not install. install with 'pip install envpool'")
    raise e


from sf_examples.envpool.envpool_wrappers import EnvPoolResetFixWrapper
from sf_examples.mujoco.mujoco_utils import MUJOCO_ENVS


def mujoco_env_by_name(name):
    for cfg in MUJOCO_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Mujoco env")


def make_mujoco_env(env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    assert cfg.batched_sampling, "batched sampling must be used when using envpool"
    # assert cfg.num_envs_per_worker == 1, "when using envpool, set num_envs_per_worker=1 and use --env_agents="
    mujoco_spec = mujoco_env_by_name(env_name)
    env_kwargs = dict()
    if env_config is not None:
        env_kwargs["seed"] = env_config.env_id

    log.debug(
        f"Envpool uses {cfg.envpool_num_threads} threads and thread affinity offset {cfg.envpool_thread_affinity_offset}"
    )

    env = envpool.make(
        mujoco_spec.env_id,
        env_type="gym",
        num_envs=cfg.env_agents,
        batch_size=cfg.env_agents,  # step all agents at the same time
        num_threads=cfg.envpool_num_threads,  # defaults to batch_size == num_envs which is not what we want (will create waay too many threads)
        thread_affinity_offset=cfg.envpool_thread_affinity_offset,
        **env_kwargs,
    )
    env = EnvPoolResetFixWrapper(env)
    env.num_agents = cfg.env_agents
    return env
