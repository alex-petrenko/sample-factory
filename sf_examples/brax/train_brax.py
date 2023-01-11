"""
Brax env integration.
"""

import sys
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.utils.dlpack as tpack
from brax.envs import to_torch
from gym.core import RenderFrame
from torch import Tensor

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, Env
from sample_factory.utils.utils import log


def jax_to_torch(tensor):
    from jax._src.dlpack import to_dlpack

    tensor = to_dlpack(tensor)
    tensor = tpack.from_dlpack(tensor)
    return tensor


def torch_to_jax(tensor):
    from jax._src.dlpack import from_dlpack

    tensor = tpack.to_dlpack(tensor)
    tensor = from_dlpack(tensor)
    return tensor


class BraxEnv(gym.Env):
    # noinspection PyProtectedMember
    def __init__(self, brax_env, num_actors):
        self.env = brax_env
        self.num_agents = num_actors
        self.env.closed = False
        self.env.viewer = None

        self.truncated = None

        observation_size = self.env.observation_space.shape[1]
        action_size = self.env.action_space.shape[1]

        obs_high = np.inf * np.ones(observation_size)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        action_high = np.ones(action_size)
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

    def step(self, action):
        action = torch_to_jax(action)
        next_obs, reward, terminated, info = self.env.step(action)
        next_obs = jax_to_torch(next_obs)
        reward = jax_to_torch(reward)
        terminated = jax_to_torch(terminated).to(torch.bool)
        return next_obs, reward, terminated, self.truncated, info

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict]:
        log.debug(f"Resetting env {self.env} with {self.num_agents} parallel agents...")
        obs = self.env.reset()
        obs = jax_to_torch(obs)
        self.truncated = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        log.debug(f"reset() done, obs.shape={obs.shape}!")
        return obs, {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass


def make_brax_env(full_env_name: str, cfg: Config, _env_config=None, render_mode: Optional[str] = None) -> Env:
    assert (
        full_env_name in env_configs.keys()
    ), f"Env {full_env_name} is not supported. Supported envs: {list(env_configs.keys())}"

    from brax import envs

    gym_env = envs.create_gym_env(env_name=full_env_name, batch_size=cfg.env_agents, seed=0, backend="gpu")
    # gym_env = to_torch.JaxToTorchWrapper(gym_env, device="cuda")
    env = BraxEnv(gym_env, cfg.env_agents)
    return env


def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument(
        "--env_agents",
        default=2048,
        type=int,
        help="Num. agents in a vectorized env",
    )


def override_default_params_func(env, parser):
    """Most of these parameters are the same as IsaacGymEnvs default config files."""

    parser.set_defaults(
        # we're using a single very vectorized env, no need to parallelize it further
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],  # obviously need a GPU
        train_for_env_steps=100000000,
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization="torch_default",
        env_gpu_actions=True,
        reward_scale=0.01,
        max_grad_norm=1.0,
        rollout=16,
        batch_size=32768,
        num_batches_per_epoch=1,
        num_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.0,
        nonlinearity="elu",
        encoder_mlp_layers=[256, 128, 64],
        learning_rate=3e-4,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.008,
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=False,  # I think Brax does not return truncation info
        normalize_input=True,
        normalize_returns=True,
        save_best_after=int(5e6),
        serial_mode=True,
        async_rl=False,
        # use_env_info_cache=True,  # speeds up startup
    )

    # override default config parameters for specific envs
    if env in env_configs:
        parser.set_defaults(**env_configs[env])


# custom default configuration parameters for specific envs
# add more envs here analogously (env names should match config file names in IGE)
env_configs = dict(
    ant=dict(
        encoder_mlp_layers=[256, 128, 64],
        experiment_summaries_interval=3,  # experiments are short so we should save summaries often
        save_every_sec=15,
    ),
    humanoid=dict(
        encoder_mlp_layers=[512, 256, 128],
    ),
    halfcheetah=dict(
        encoder_mlp_layers=[256, 128, 64],
    ),
    walker2d=dict(
        encoder_mlp_layers=[256, 128, 64],
    ),
)


def register_brax_custom_components():
    for env_name in env_configs:
        register_env(env_name, make_brax_env)


def parse_brax_cfg(evaluation=False):
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params_func(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser)
    return final_cfg


def main():
    """Script entry point."""
    register_brax_custom_components()
    cfg = parse_brax_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
