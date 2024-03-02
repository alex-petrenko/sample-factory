from typing import Optional

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

from sample_factory.algo.utils.gymnasium_utils import patch_non_gymnasium_env
from sf_examples.nethack.utils.wrappers import (
    BlstatsInfoWrapper,
    GymV21CompatibilityV0,
    NLETimeLimit,
    PrevActionsWrapper,
    RenderCharImagesWithNumpyWrapperV2,
    SeedActionSpaceWrapper,
    TaskRewardsInfoWrapper,
)

NETHACK_ENVS = dict(
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
    assert render_mode in (None, "human", "full", "ansi", "string", "rgb_array")

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
    )
    if env_name == "challenge":
        kwargs["no_progress_timeout"] = 150

    if env_name in ("staircase", "pet", "oracle"):
        kwargs.update(reward_win=cfg.reward_win, reward_lose=cfg.reward_lose)
    # else:  # print warning once
    # warnings.warn("Ignoring cfg.reward_win and cfg.reward_lose")

    env = env_class(**kwargs)

    if cfg.add_image_observation:
        env = RenderCharImagesWithNumpyWrapperV2(
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

    # convert gym env to gymnasium one, due to issues with render NLE in reset
    gymnasium_env = GymV21CompatibilityV0(env=env)

    # preserving potential multi-agent env attributes
    if hasattr(env, "num_agents"):
        gymnasium_env.num_agents = env.num_agents
    if hasattr(env, "is_multiagent"):
        gymnasium_env.is_multiagent = env.is_multiagent
    env = gymnasium_env

    env = patch_non_gymnasium_env(env)

    if render_mode:
        env.render_mode = render_mode

    if cfg.serial_mode and cfg.num_workers == 1:
        # full reproducability can only be achieved in serial mode and when there is only 1 worker
        env = SeedActionSpaceWrapper(env)

    return env
