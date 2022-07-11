from typing import Optional, Tuple, Union, List
import numpy as np
import gym
import sys

from sample_factory.algo.utils.context import global_env_registry
from sample_factory.cfg.arguments import parse_args
from sample_factory.train import run_rl

class IdentityEnvMixedActions(gym.Env):
    def __init__(self, size=4):
        self.observation_space = gym.spaces.Box(-1,1,shape=(size,))
        self._observation_space = gym.spaces.Discrete(size)

        self.action_space =  gym.spaces.Tuple(
            [gym.spaces.Discrete(size), gym.spaces.Box(-1,1,shape=(size,))]
            )
        self.ep_length = 10
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.eps = 0.05
        self.reset()

    def reset(self):
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state

    def _get_reward(self, action: Union[int, np.ndarray]) -> float:
        discrete_reward = 1.0 if np.argmax(self.state)  == action[0] else 0.0
        continuous_reward = 1.0 if (np.argmax(self.state) - self.eps) <= sum(action[1]) <= (np.argmax(self.state) + self.eps) else 0.0

        return discrete_reward + continuous_reward

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

def override_defaults(env, parser):
    parser.set_defaults(
        batched_sampling=False,
        num_workers=4,
        num_envs_per_worker=4,
        worker_num_splits=2,
        train_for_env_steps=10000,

        encoder_type='mlp',
        encoder_subtype='mlp_mujoco',
        hidden_size=64,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
        nonlinearity='tanh',
        batch_size=1024,
    )

def make_env(env_name, cfg, **kwargs):
    return IdentityEnvMixedActions(4)

def register_test_components():
    global_env_registry().register_env(
        env_name_prefix='non_batched_mix_dist_env',
        make_env_func=make_env,
        add_extra_params_func=None,
        override_default_params_func=override_defaults
    )    


def test_non_batched_mixed_action_dists():
    """Script entry point."""
    register_test_components()
    cfg = parse_args(argv=["--algo=APPO", "--env=non_batched_mix_dist_env", f"--experiment=test_non_batched_mixed_action_dists"])
    status = run_rl(cfg)
    return status

