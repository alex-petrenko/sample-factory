from typing import Optional

try:
    import envpool
except ImportError as e:
    print(e)
    print("Trying to import envpool when it is not install. install with 'pip install envpool'")

import gym

from sample_factory.utils.utils import is_module_available
from sf_examples.envpool_examples.envpool_wrappers import EnvPoolTo5Tuple


def mujoco_available():
    return is_module_available("mujoco")


class MujocoSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


MUJOCO_ENVS = [
    MujocoSpec("mujoco_hopper", "Hopper-v4"),
    MujocoSpec("mujoco_halfcheetah", "HalfCheetah-v4"),
    MujocoSpec("mujoco_humanoid", "Humanoid-v4"),
    MujocoSpec("mujoco_ant", "Ant-v4"),
    MujocoSpec("mujoco_standup", "HumanoidStandup-v4"),
    MujocoSpec("mujoco_doublependulum", "InvertedDoublePendulum-v4"),
    MujocoSpec("mujoco_pendulum", "InvertedPendulum-v4"),
    MujocoSpec("mujoco_reacher", "Reacher-v4"),
    MujocoSpec("mujoco_walker", "Walker2d-v4"),
    MujocoSpec("mujoco_pusher", "Pusher-v4"),
    MujocoSpec("mujoco_swimmer", "Swimmer-v4"),
]


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
    env = EnvPoolTo5Tuple(env)
    env.num_agents = cfg.env_agents
    return env
