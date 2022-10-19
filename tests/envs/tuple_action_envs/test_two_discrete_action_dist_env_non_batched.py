from __future__ import annotations

from typing import List, Optional, Tuple, Union

import gym
import numpy as np

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.utils.utils import debug_log_every_n

DiscreteActions = Union[List[int], Tuple[int, ...], np.ndarray]


def get_reward(action: DiscreteActions, state) -> float:
    discrete_reward1 = 1.0 if np.argmax(state) == action[0] else 0.0
    discrete_reward2 = 1.0 if np.argmax(state) * 3 == (len(state) - action[1] - 1) else 0.0
    return discrete_reward1 + discrete_reward2


class IdentityEnvTwoDiscreteActions(gym.Env):
    def __init__(self, size=4):
        self.observation_space = gym.spaces.Box(-1, 1, shape=(size,))
        self._observation_space = gym.spaces.Discrete(size)

        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(size), gym.spaces.Discrete(size * 3)])
        self.ep_length = 10
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.eps = 0.05
        self.current_step = 0
        self.reset()

    def reset(self, **kwargs):
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state, {}

    def _choose_next_state(self) -> None:
        state = np.zeros(self.observation_space.shape)
        index = self._observation_space.sample()
        state[index] = 1.0
        self.state = state

    def step(self, action: DiscreteActions):
        debug_log_every_n(1000, f"{action=}, {type(action)=}, {type(action[0])=}")
        assert isinstance(action[0], (int, np.integer))
        assert isinstance(action[1], (int, np.integer))
        reward = get_reward(action, self.state)
        self._choose_next_state()
        self.current_step += 1
        terminated = truncated = self.current_step >= self.ep_length
        return self.state, reward, terminated, truncated, {}

    def close(self):
        pass

    def render(self):
        pass


def override_defaults(parser):
    parser.set_defaults(
        batched_sampling=False,
        num_workers=4,
        num_envs_per_worker=4,
        worker_num_splits=2,
        train_for_env_steps=10000,
        env_frameskip=1,
        nonlinearity="tanh",
        batch_size=1024,
        decorrelate_envs_on_one_worker=False,
    )


def make_env(_env_name, _cfg, _cfg_env, render_mode: Optional[str] = None):
    return IdentityEnvTwoDiscreteActions(4)


def register_test_components():
    register_env(
        "non_batched_two_discrete_dist_env",
        make_env,
    )


def test_non_batched_two_discrete_action_dists():
    """Script entry point."""
    register_test_components()

    argv = [
        "--algo=APPO",
        "--env=non_batched_two_discrete_dist_env",
        "--experiment=test_non_batched_two_discrete_dists",
        "--restart_behavior=overwrite",
        "--device=cpu",
    ]

    parser, cfg = parse_sf_args(argv=argv)
    override_defaults(parser)
    cfg = parse_full_cfg(parser, argv=argv)

    status = run_rl(cfg)
    return status
