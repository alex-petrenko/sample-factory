from typing import Optional


class EpsilonSchedule:
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        decay_steps: int = 1_000_000,
    ):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self._step = 0

    def get_epsilon(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step

        if step >= self.decay_steps:
            return self.epsilon_end

        # Linear interpolation
        progress = step / self.decay_steps
        epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
        return epsilon

    def step(self, n: int = 1) -> float:
        """
        Advance the schedule and return the current epsilon.

        Args:
            n: Number of steps to advance.

        Returns:
            Current epsilon value.
        """
        epsilon = self.get_epsilon()
        self._step += n
        return epsilon

    @property
    def current_epsilon(self) -> float:
        """Get current epsilon without advancing."""
        return self.get_epsilon()

    def __repr__(self) -> str:
        return (
            f"EpsilonSchedule(start={self.epsilon_start}, end={self.epsilon_end}, "
            f"decay_steps={self.decay_steps}, current_step={self._step})"
        )
