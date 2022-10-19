from typing import Optional

import gym

from sample_factory.utils.utils import is_module_available


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


def make_mujoco_env(env_name, _cfg, _env_config, render_mode: Optional[str] = None, **kwargs):
    mujoco_spec = mujoco_env_by_name(env_name)
    env = gym.make(mujoco_spec.env_id, render_mode=render_mode)
    return env
