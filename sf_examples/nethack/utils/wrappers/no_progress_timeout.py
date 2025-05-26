import gymnasium as gym
from nle import nethack


class NoProgressTimeout(gym.Wrapper):
    def __init__(self, env, no_progress_timeout: int = 150):
        super().__init__(env)
        self.no_progress_timeout = no_progress_timeout
        self.env._check_abort = self._check_abort

    def reset(self, *args, **kwargs):
        self._turns = None
        self._no_progress_count = 0
        return super().reset(*args, **kwargs)

    def _check_abort(self, observation):
        """Check if time has stopped and no observations has changed long enough
        to trigger an abort."""

        turns = observation[self.env._blstats_index][nethack.NLE_BL_TIME]
        if self._turns == turns:
            self._no_progress_count += 1
        else:
            self._turns = turns
            self._no_progress_count = 0
        return self.env._steps >= self.env._max_episode_steps or self._no_progress_count >= self.no_progress_timeout
