from typing import Optional

import gymnasium as gym
import nle  # noqa: F401
from nle import nethack

from sample_factory.utils.utils import is_module_available
from sf_examples.nethack.utils.wrappers import (
    BlstatsInfoWrapper,
    NoProgressTimeout,
    PrevActionsWrapper,
    TaskRewardsInfoWrapper,
    TileTTY,
)


def nethack_available():
    return is_module_available("nle")


class NetHackSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


NETHACK_ENVS = [
    NetHackSpec("nethack_staircase", "NetHackStaircase-v0"),
    NetHackSpec("nethack_score", "NetHackScore-v0"),
    NetHackSpec("nethack_pet", "NetHackStaircasePet-v0"),
    NetHackSpec("nethack_oracle", "NetHackOracle-v0"),
    NetHackSpec("nethack_gold", "NetHackGold-v0"),
    NetHackSpec("nethack_eat", "NetHackEat-v0"),
    NetHackSpec("nethack_scout", "NetHackScout-v0"),
    NetHackSpec("nethack_challenge", "NetHackChallenge-v0"),
]


def nethack_env_by_name(name):
    for cfg in NETHACK_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown NetHack env")


def make_nethack_env(env_name, cfg, _env_config, render_mode: Optional[str] = None, **kwargs):
    nethack_spec = nethack_env_by_name(env_name)

    observation_keys = (
        "message",
        "blstats",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        # ALSO AVAILABLE (OFF for speed)
        # "specials",
        # "colors",
        # "chars",
        # "glyphs",
        # "inv_glyphs",
        # "inv_strs",
        # "inv_letters",
        # "inv_oclasses",
    )

    kwargs = dict(
        character=cfg.character,
        max_episode_steps=cfg.max_episode_steps,
        observation_keys=observation_keys,
        penalty_step=cfg.penalty_step,
        penalty_time=cfg.penalty_time,
        penalty_mode=cfg.fn_penalty_step,
        savedir=cfg.savedir,
        save_ttyrec_every=cfg.save_ttyrec_every,
        allow_all_yn_questions=True,
        allow_all_modes=True,
    )

    if env_name in ("nethack_staircase", "nethack_pet", "nethack_oracle"):
        kwargs.update(reward_win=cfg.reward_win, reward_lose=cfg.reward_lose)
    if env_name != "nethack_challenge":
        kwargs.update(actions=nethack.ACTIONS)
    # else:  # print warning once
    # warnings.warn("Ignoring cfg.reward_win and cfg.reward_lose")

    env = gym.make(nethack_spec.env_id, render_mode=render_mode, **kwargs)

    env = NoProgressTimeout(env, no_progress_timeout=150)

    if cfg.add_image_observation:
        env = TileTTY(
            env,
            crop_size=cfg.crop_dim,
            rescale_font_size=(cfg.pixel_size, cfg.pixel_size),
        )

    if cfg.use_prev_action:
        env = PrevActionsWrapper(env)

    if cfg.add_stats_to_info:
        env = BlstatsInfoWrapper(env)
        env = TaskRewardsInfoWrapper(env)

    return env
