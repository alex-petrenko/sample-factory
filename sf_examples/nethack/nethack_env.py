from typing import Optional

from nle import nethack
from nle.env.tasks import (
    NetHackChallenge,
    NetHackEat,
    NetHackGold,
    NetHackOracle,
    NetHackScore,
    NetHackScout,
    NetHackStaircase,
    NetHackStaircasePet,
)
from nle_progress.task import NetHackProgress

from sf_examples.nethack.utils.wrappers import (
    BlstatsInfoWrapper,
    GymV21CompatibilityV0,
    NLETimeLimit,
    NoProgressTimeout,
    PrevActionsWrapper,
    TaskRewardsInfoWrapper,
    TileTTY,
)

NETHACK_ENVS = dict(
    nethack_progress=NetHackProgress,
    nethack_staircase=NetHackStaircase,
    nethack_score=NetHackScore,
    nethack_pet=NetHackStaircasePet,
    nethack_oracle=NetHackOracle,
    nethack_gold=NetHackGold,
    nethack_eat=NetHackEat,
    nethack_scout=NetHackScout,
    nethack_challenge=NetHackChallenge,
)


def nethack_env_by_name(name):
    if name in NETHACK_ENVS.keys():
        return NETHACK_ENVS[name]
    else:
        raise Exception("Unknown NetHack env")


def make_nethack_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    env_class = nethack_env_by_name(env_name)

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
        actions=nethack.ACTIONS,
    )

    if env_name in ("staircase", "pet", "oracle"):
        kwargs.update(reward_win=cfg.reward_win, reward_lose=cfg.reward_lose)
    # else:  # print warning once
    # warnings.warn("Ignoring cfg.reward_win and cfg.reward_lose")

    env = env_class(**kwargs)
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

    # add TimeLimit.truncated to info
    env = NLETimeLimit(env)

    # convert gym env to gymnasium one
    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
