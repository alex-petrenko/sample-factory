from __future__ import annotations

from typing import List, Optional

import gym
import numpy as np

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from tests.envs.tuple_action_envs.test_two_discrete_action_dist_env_non_batched import DiscreteActions, get_reward


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
        reward = get_reward(action, self.state)
        self._choose_next_state()
        self.current_step += 1
        terminated = truncated = self.current_step >= self.ep_length
        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass


class BatchedIdentityEnvTwoDiscreteActions(gym.Env):
    def __init__(self, size=4) -> None:
        n_envs = 4
        self.envs = [IdentityEnvTwoDiscreteActions(size) for _ in range(n_envs)]
        self.num_agents = n_envs

        super().__init__()

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space

    def reset(self, **kwargs):
        obss = []
        infos = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset()
            obss.append(obs)
            infos.append(info)
        return obss, infos

    def step(self, action: List[DiscreteActions] | np.ndarray):
        obss, rewards, terms, truncs, infos = [], [], [], [], []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step([action[i][0], action[i][1]])
            obss.append(obs),
            rewards.append(reward)
            terms.append(terminated)
            truncs.append(truncated)
            infos.append(info)

        return obss, rewards, terms, truncs, infos

    def render(self):
        pass

    def close(self):
        pass


def override_defaults(parser):
    parser.set_defaults(
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        train_for_env_steps=10000,
        nonlinearity="tanh",
        batch_size=1024,
        decorrelate_envs_on_one_worker=False,
    )


def make_env(_env_name, _cfg, _cfg_env, render_mode: Optional[str] = None):
    return BatchedIdentityEnvTwoDiscreteActions(4)


def register_test_components():
    register_env(
        "batched_two_discrete_dist_env",
        make_env,
    )


def test_batched_two_discrete_action_dists():
    """Script entry point."""
    register_test_components()
    argv = [
        "--algo=APPO",
        "--env=batched_two_discrete_dist_env",
        "--experiment=test_batched_two_discrete_dists",
        "--restart_behavior=overwrite",
        "--device=cpu",
    ]
    parser, cfg = parse_sf_args(argv=argv)

    override_defaults(parser)
    cfg = parse_full_cfg(parser, argv=argv)

    status = run_rl(cfg)
    return status
