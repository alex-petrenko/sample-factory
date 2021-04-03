import math
from collections import deque
import numpy as np

import gym


class ExplorationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.landmarks = deque([], maxlen=200)
        self.landmark_threshold = 75.0

    def _calc_intrinsic_reward(self, info):
        pos = info.get('pos', None)
        if pos is None:
            return 0.0

        x, y, angle = pos['agent_x'], pos['agent_y'], pos['agent_a']
        reward = 0.0

        # check if we are far from all known landmarks
        is_new_landmark = True
        for landmark in self.landmarks:
            x_old, y_old, angle_old = landmark
            distance = math.sqrt((x - x_old) ** 2 + (y - y_old) ** 2)

            angle_diff = abs(angle - angle_old)
            angle_diff = min(angle_diff, 360.0 - angle_diff)
            distance += angle_diff / 2

            if distance < self.landmark_threshold:
                is_new_landmark = False
                break

        if is_new_landmark:
            self.landmarks.appendleft((x, y, angle))
            reward += 0.1
            # log.debug('New landmark %.3f %.3f %.3f rew %.3f, num %d', x, y, angle, reward, len(self.landmarks))

        while len(self.landmarks) >= self.landmarks.maxlen:
            random_idx = np.random.randint(0, len(self.landmarks))
            # log.info('Delete landmark %d', random_idx)
            del self.landmarks[random_idx]

        return reward

    def reset(self, **kwargs):
        self.landmarks = deque([], maxlen=200)
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        exploration_reward = self._calc_intrinsic_reward(info)
        info['intrinsic_reward'] = info.get('intrinsic_reward', 0.0) + exploration_reward
        return obs, reward, done, info
