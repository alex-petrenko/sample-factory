import cv2
import numpy as np


def key_to_action(key):
    from pynput.keyboard import Key

    action_table = {
        Key.up: 0,
        Key.down: 1,
        Key.right: 4,
        Key.left: 5,
        Key.alt: 6,
        Key.ctrl: 8,
        Key.shift: 9,
    }

    return action_table.get(key, None)


_DOOM_WINDOWS = set()


def concat_grid(obs):
    obs = [cvt_doom_obs(o) for o in obs]

    max_horizontal = 4
    horizontal = min(max_horizontal, len(obs))

    while len(obs) % horizontal != 0:
        obs.append(np.zeros_like(obs[0]))  # pad with black images until right size

    vertical = max(1, len(obs) // horizontal)

    assert vertical * horizontal == len(obs)

    obs_concat_h = []
    for i in range(vertical):
        obs_concat_h.append(np.concatenate(obs[i * horizontal:(i + 1) * horizontal], axis=1))

    obs_concat = np.concatenate(obs_concat_h, axis=0)
    return obs_concat


def cvt_doom_obs(obs):
    w, h = 400, 225
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    obs = cv2.resize(obs, (w, h))
    return obs
