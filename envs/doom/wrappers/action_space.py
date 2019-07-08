import gym
from gym.spaces import Discrete


def doom_action_space():
    """
    Standard action space for full-featured Doom environments (e.g. deathmatch).
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
        SELECT_NEXT_WEAPON
        SELECT_PREV_WEAPON
        ATTACK
        SPEED
    """
    return gym.spaces.Tuple((
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, move right, move left
        Discrete(3),  # noop, turn right, turn left
        Discrete(3),  # noop, prev_weapon, next_weapon
        Discrete(2),  # noop, attack
        Discrete(2),  # noop, sprint
    ))
