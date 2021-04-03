import math


class LinearDecay:
    def __init__(self, milestones, staircase=None):
        """
        Linear decay of some value according to schedule.
        See tests for usage sample_factory_examples.

        :param milestones: list
        List of tuples (step, desired_value)
        E.g. [(0, 100), (1000, 50)] means for step <= 0 use value 100, between step 0 and 1000 interpolate the value
        between 100 and 50, then keep at 50 forever.
        :param staircase: int
        If None then no rounding is applied.
        If int then the value will move one "stair" at a time (if staircase=10, then value will be: 100, 90, 80, ...)

        """
        if len(milestones) == 0:
            raise Exception('Milestones list should not be empty!')

        self._schedule = sorted(milestones)
        self._staircase = staircase

    def at(self, step):
        if step <= self._schedule[0][0]:
            return self._schedule[0][1]
        if step >= self._schedule[-1][0]:
            return self._schedule[-1][1]

        # find where we are in terms of milestones
        milestone = 0
        while self._schedule[milestone][0] < step:
            milestone += 1

        x = step
        x0, y0 = self._schedule[milestone - 1]
        x1, y1 = self._schedule[milestone]

        # linear interpolation
        value = y0 * (1 - (x-x0)/(x1-x0)) + y1 * (1 - (x1-x)/(x1-x0))

        if self._staircase is None:
            return value
        else:
            num_stairs = math.floor(value / self._staircase)
            return max(num_stairs * self._staircase, self._schedule[0][1])
