import sys

import gym


def main():
    env = gym.make('HalfCheetah-v2')
    print("Observation space's shape: ", str(env.observation_space.low.shape))
    print("Action space's shape: ", str(env.action_space.low.shape))
    print(1)

if __name__ == '__main__':
    sys.exit(main())

