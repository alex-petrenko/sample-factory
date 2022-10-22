from typing import Optional

try:
    import envpool
except ImportError as e:
    print(e)
    print("Trying to import envpool when it is not install. install with 'pip install envpool'")


from sf_examples.envpool.envpool_wrappers import EnvPoolResetFixWrapper
from sf_examples.mujoco.mujoco_utils import MUJOCO_ENVS


def mujoco_env_by_name(name):
    for cfg in MUJOCO_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Mujoco env")


def make_mujoco_env(env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    assert cfg.batched_sampling, "batched sampling must be used when using envpool"
    assert cfg.num_envs_per_worker == 1, "when using envpool, set num_envs_per_worker=1 and use --env_agents="
    mujoco_spec = mujoco_env_by_name(env_name)
    env_kwargs = dict()
    if env_config is not None:
        env_kwargs["seed"] = env_config.env_id
    env = envpool.make(mujoco_spec.env_id, env_type="gym", num_envs=cfg.env_agents, **env_kwargs)
    env = EnvPoolResetFixWrapper(env)
    env.num_agents = cfg.env_agents
    return env
