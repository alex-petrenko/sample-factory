import gym
from gym.spaces import Discrete, Box

import numpy as np

from sample_factory.algorithms.utils.spaces.discretized import Discretized


def key_to_action_basic(key):
    from pynput.keyboard import Key
    table = {Key.left: 0, Key.right: 1, Key.up: 2, Key.down: 3}
    return table.get(key, None)


def doom_turn_and_attack_only():
    """
    TURN_LEFT
    TURN_RIGHT
    ATTACK
    """
    space = gym.spaces.Tuple((
        Discrete(3),  # noop, turn left, turn right
        Discrete(2),  # noop, attack
    ))

    return space


def doom_action_space_basic():
    """
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    MOVE_BACKWARD
    """
    space = gym.spaces.Tuple((
        Discrete(3),  # noop, turn left, turn right
        Discrete(3),  # noop, forward, backward
    ))

    space.key_to_action = key_to_action_basic
    return space


def doom_action_space_extended():
    """
    This function assumes the following list of available buttons:
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    MOVE_BACKWARD
    MOVE_LEFT
    MOVE_RIGHT
    ATTACK
    """
    space = gym.spaces.Tuple((
        Discrete(3),  # noop, turn left, turn right
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, strafe left, strafe right
        Discrete(2),  # noop, attack
    ))

    return space


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
        SELECT_NEXT_WEAPON
        SELECT_PREV_WEAPON
        ATTACK
        SPEED
        TURN_LEFT_RIGHT_DELTA
    """
    return gym.spaces.Tuple((
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, move right, move left
        Discrete(3),  # noop, prev_weapon, next_weapon
        Discrete(2),  # noop, attack
        Discrete(2),  # noop, sprint
        Box(np.float32(-1.0), np.float32(1.0), (1,)),
    ))


def doom_action_space_discretized():
    return gym.spaces.Tuple((
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, move right, move left
        Discrete(3),  # noop, prev_weapon, next_weapon
        Discrete(2),  # noop, attack
        Discrete(2),  # noop, sprint
        Discretized(11, min_action=-10.0, max_action=10.0),  # turning using discretized continuous control
    ))


def doom_action_space_discretized_no_weap():
    return gym.spaces.Tuple((
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, move right, move left
        Discrete(2),  # noop, attack
        Discrete(2),  # noop, sprint
        Discretized(11, min_action=-10.0, max_action=10.0),  # turning using discretized continuous control
    ))


def doom_action_space_continuous_no_weap():
    return gym.spaces.Tuple((
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, move right, move left
        Discrete(2),  # noop, attack
        Discrete(2),  # noop, sprint
        Box(float(-1.0), float(1.0), (1,)),
    ))


def doom_action_space_discrete():
    return gym.spaces.Tuple((
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, move right, move left
        Discrete(3),  # noop, turn right, turn left
        Discrete(3),  # noop, prev_weapon, next_weapon
        Discrete(2),  # noop, attack
        Discrete(2),  # noop, sprint
    ))


def doom_action_space_discrete_no_weap():
    return gym.spaces.Tuple((
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, move right, move left
        Discrete(3),  # noop, turn right, turn left
        Discrete(2),  # noop, attack
        Discrete(2),  # noop, sprint
    ))


def doom_action_space_full_discretized(with_use=False):
    """
        MOVE_FORWARD
        MOVE_BACKWARD
        MOVE_RIGHT
        MOVE_LEFT
        SELECT_WEAPON1
        SELECT_WEAPON2
        SELECT_WEAPON3
        SELECT_WEAPON4
        SELECT_WEAPON5
        SELECT_WEAPON6
        SELECT_WEAPON7
        ATTACK
        SPEED
        TURN_LEFT_RIGHT_DELTA
    """
    spaces = [
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, move right, move left
        Discrete(8),  # noop, select weapons 1-7
        Discrete(2),  # noop, attack
        Discrete(2),  # noop, sprint
    ]
    if with_use:
        spaces.append(Discrete(2))  # noop, use

    spaces.append(Discretized(21, min_action=-12.5, max_action=12.5))  # turning using discretized continuous control

    return gym.spaces.Tuple(spaces)
