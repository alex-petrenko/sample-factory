import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from sample_factory.envs.env_wrappers import RecordingWrapper


class _FrameSequenceEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(2, 2, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, observations):
        self._observations = iter(observations)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), {}

    def step(self, action):
        return next(self._observations), 1.0, False, False, {}


@pytest.mark.parametrize("missing_frame", [None, np.empty((0, 0, 3), dtype=np.uint8)])
def test_recording_wrapper_skips_missing_frames(tmp_path: Path, missing_frame):
    valid_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    env = RecordingWrapper(_FrameSequenceEnv([missing_frame, valid_frame]), str(tmp_path), player_id=0)
    env.reset()

    env.step(0)

    episode_dir = tmp_path / "ep_000_p0"
    assert list(episode_dir.glob("*.png")) == []

    env.step(1)
    env.reset()

    completed_episode_dir = tmp_path / "ep_000_p0_r2.00"
    assert [path.name for path in completed_episode_dir.glob("*.png")] == ["00000.png"]
    assert json.loads((completed_episode_dir / "actions.json").read_text()) == [0, 1]
