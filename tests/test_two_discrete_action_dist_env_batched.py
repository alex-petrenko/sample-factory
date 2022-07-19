import sys
from typing import List, Optional, Tuple, Union

import gym
import numpy as np

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl


class IdentityEnvTwoDiscreteActions(gym.Env):
    def __init__(self, size=4):
        self.observation_space = gym.spaces.Box(-1, 1, shape=(size,))
        self._observation_space = gym.spaces.Discrete(size)

        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(size), gym.spaces.Discrete(size * 3)])
        self.ep_length = 10
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.eps = 0.05
        self.reset()

    def reset(self):
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state

    def _get_reward(self, action: Union[int, int]) -> float:
        discrete_reward1 = 1.0 if np.argmax(self.state) == action[0] else 0.0
        discrete_reward2 = 1.0 if np.argmax(self.state) * 3 == (len(self.state) - action[1] - 1) else 0.0

        return discrete_reward1 + discrete_reward2

    def _choose_next_state(self) -> None:
        state = np.zeros(self.observation_space.shape)
        index = self._observation_space.sample()
        state[index] = 1.0
        self.state = state

    def step(self, action: List[np.ndarray]):
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def close(self):
        pass

    def seed(self, value):
        return


class BatchedIdentityEnvTwoDiscreteActionss(gym.Env):
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

    def reset(self):
        obss = []
        for i, env in enumerate(self.envs):
            obs = env.reset()
            obss.append(obs)
        return obss

    def step(self, action: List[np.ndarray]):
        obss, rewards, dones, infos = [], [], [], []

        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step([action[i][0], action[i][1]])
            obss.append(obs),
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return obss, rewards, dones, infos

    def close(self):
        pass

    def seed(self, value):
        return


def override_defaults(parser):
    parser.set_defaults(
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        train_for_env_steps=10000,
        encoder_type="mlp",
        encoder_subtype="mlp_mujoco",
        hidden_size=64,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
        nonlinearity="tanh",
        batch_size=1024,
    )


def make_env(env_name, cfg, cfg_env):
    return BatchedIdentityEnvTwoDiscreteActionss(4)


def register_test_components():
    register_env(
        "batched_two_discete_dist_env",
        make_env,
    )


def test_batched_two_discrete_action_dists():
    """Script entry point."""
    register_test_components()
    argv = ["--algo=APPO", "--env=batched_two_discete_dist_env", "--experiment=test_batched_two_discete_dists"]
    parser, cfg = parse_sf_args(argv=argv)

    override_defaults(parser)
    cfg = parse_full_cfg(parser, argv=argv)

    status = run_rl(cfg)
    return status
