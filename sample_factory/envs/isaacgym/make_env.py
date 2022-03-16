import os.path
from os.path import join
from typing import Optional, Union, Tuple

import gym
import yaml
from gym.core import ObsType, ActType

from sample_factory.envs.isaacgym.isaacgymenvs.tasks.ant import AntRun

isaacgym_task_map = {
    "AntRunSF": AntRun,
    # "HumanoidRun": HumanoidRun,
    # "ShadowHand": ShadowHand,
    # "AllegroHand": AllegroHand,
    # "AllegroKuka": resolve_allegro_kuka,
    # "Trifinger": Trifinger,
    # #"Ingenuity": Ingenuity,
    # "Anymal": Anymal,
    # "AnymalTerrain": resolve_anymal_terrain,
}


class IsaacGymVecEnv(gym.Env):
    def __init__(self, isaacgym_env):
        self.env = isaacgym_env
        self.num_agents = self.env.num_envs  # TODO: what about vectorized multi-agent envs? should we take num_agents into account also?

        self.action_space = self.env.action_space

        # isaacgym environments actually return dicts
        self.observation_space = gym.spaces.Dict(dict(obs=self.env.observation_space))

    def reset(self, *args, **kwargs):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def render(self, mode='human'):
        pass


def make_isaacgym_env(full_env_name, cfg=None, env_config=None):
    task_name = '_'.join(full_env_name.split('_')[1:])

    # import isaacgymenvs
    # module_dir = isaacgymenvs.__path__[0]
    # cfg_dir = join(module_dir, 'cfg')

    cfg_dir = join(os.path.dirname(os.path.realpath(__file__)), 'cfg')
    cfg_file = join(cfg_dir, 'task', f'{task_name}.yaml')

    with open(cfg_file, 'r') as yaml_stream:
        task_cfg = yaml.safe_load(yaml_stream)

    sim_device = 'cuda:0' if cfg.actor_worker_gpus else 'cpu'  # TODO: better logic!

    task_cfg['env']['numEnvs'] = cfg.env_agents

    env = isaacgym_task_map[task_cfg['name']](
        cfg=task_cfg,
        sim_device=sim_device,
        headless=cfg.env_headless,
    )

    env = IsaacGymVecEnv(env)
    return env
