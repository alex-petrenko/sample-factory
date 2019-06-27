import gym
from gym.spaces import Discrete


def doom_action_space():
    """
    Standard action space for full-featured Doom environments (e.g. deathmatch).
    TODO: weapon change?
    TODO: crouch?
    TODO: strafe?

    This should precisely correspond to the available_buttons configuration in the .cfg file.
    This function assumes:
        MOVE_FORWARD
        MOVE_BACKWARD
        MOVE_RIGHT
        MOVE_LEFT
        TURN_RIGHT
        TURN_LEFT
        ATTACK
        SPEED
    """
    return gym.spaces.Tuple((
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, move right, move left
        Discrete(3),  # noop, turn right, turn left
        Discrete(2),  # noop, attack
        Discrete(2),  # noop, sprint
    ))



class DoomActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.Tuple((env.action_space, gym.spaces.Discrete(2)))

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action[0])
