from gym.spaces import Discrete

from envs.doom.action_space import doom_action_space, doom_action_space_no_weap, doom_action_space_discrete, \
    doom_action_space_hybrid, doom_action_space_hybrid_no_weap, doom_action_space_experimental
from envs.doom.doom_gym import VizdoomEnv
from envs.doom.wrappers.additional_input import DoomAdditionalInputAndRewards
from envs.doom.wrappers.bot_difficulty import BotDifficultyWrapper
from envs.doom.wrappers.multiplayer_stats import MultiplayerStatsWrapper
from envs.doom.wrappers.observation_space import SetResolutionWrapper, resolutions
from envs.doom.wrappers.scenario_wrappers.gathering_reward_shaping import DoomGatheringRewardShaping
from envs.env_wrappers import ResizeWrapper, RewardScalingWrapper, TimeLimitWrapper, RecordingWrapper, \
    PixelFormatChwWrapper


DOOM_W = 128
DOOM_H = 72


class DoomSpec:
    def __init__(
            self, name, env_spec_file, action_space, reward_scaling=1.0, default_timeout=int(1e9),
            num_agents=1, num_bots=0,
            extra_wrappers=None,
    ):
        self.name = name
        self.env_spec_file = env_spec_file
        self.action_space = action_space
        self.reward_scaling = reward_scaling
        self.default_timeout = default_timeout

        # 1 for singleplayer, >1 otherwise
        self.num_agents = num_agents

        # CLI arguments override this (see enjoy_rllib.py)
        self.num_bots = num_bots

        # expect list of tuples (wrapper_cls, wrapper_kwargs)
        self.extra_wrappers = self._extra_wrappers_or_default(extra_wrappers)

    @staticmethod
    def _extra_wrappers_or_default(wrappers):
        if wrappers is None:
            return [(DoomAdditionalInputAndRewards, {})]
        else:
            return wrappers


DOOM_ENVS = [
    DoomSpec(
        'doom_basic', 'basic.cfg',
        Discrete(1 + 3),  # idle, left, right, attack
        0.01, 300,
        extra_wrappers=[(DoomAdditionalInputAndRewards, {'with_reward_shaping': False})],
    ),

    DoomSpec('doom_corridor', 'deadly_corridor.cfg', Discrete(1 + 7), 0.01, 2100),
    DoomSpec('doom_gathering', 'health_gathering.cfg', Discrete(1 + 3), 0.01, 2100),

    DoomSpec('doom_battle_discrete', 'D3_battle.cfg', Discrete(1 + 8), 1.0, 2100),
    DoomSpec('doom_battle_tuple_actions', 'D3_battle.cfg', doom_action_space_discrete(), 1.0, 2100),
    DoomSpec('doom_battle_continuous', 'D3_battle_continuous.cfg', doom_action_space_no_weap(), 1.0, 2100),
    DoomSpec('doom_battle_hybrid', 'D3_battle_continuous.cfg', doom_action_space_hybrid_no_weap(), 1.0, 2100),

    DoomSpec('doom_dm', 'cig.cfg', doom_action_space(), 1.0, int(1e9), num_agents=8),

    DoomSpec(
        'doom_two_colors_easy', 'two_colors_easy.cfg',
        Discrete(5),  # idle, left, right, forward, backward
        extra_wrappers=[
            # (DoomAdditionalInputAndRewards, {'with_reward_shaping': False}),
            (DoomGatheringRewardShaping, {}),
        ]
    ),

    DoomSpec(
        'doom_two_colors_hard', 'two_colors_hard.cfg',
        Discrete(5),  # idle, left, right, forward, backward
        extra_wrappers=[
            # (DoomAdditionalInputAndRewards, {'with_reward_shaping': False}),
            (DoomGatheringRewardShaping, {}),
        ]
    ),

    DoomSpec('doom_dwango5', 'dwango5_dm.cfg', doom_action_space(), 1.0, int(1e9), num_agents=8),

    DoomSpec(
        'doom_dwango5_bots',
        'dwango5_dm.cfg',
        doom_action_space_discrete(),
        1.0, int(1e9),
        num_agents=1, num_bots=7,
    ),

    DoomSpec(
        'doom_dwango5_bots_continuous',
        'dwango5_dm_continuous.cfg',
        doom_action_space(),
        1.0, int(1e9),
        num_agents=1, num_bots=7,
    ),

    DoomSpec(
        'doom_dwango5_bots_hybrid',
        'dwango5_dm_continuous.cfg',
        doom_action_space_hybrid(),
        1.0, int(1e9),
        num_agents=1, num_bots=7,
    ),

    DoomSpec(
        'doom_dwango5_bots_experimental',
        'dwango5_dm_continuous_weap.cfg',
        doom_action_space_experimental(),
        1.0, int(1e9),
        num_agents=1, num_bots=7,
    ),
]


def doom_env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Doom env')


# noinspection PyUnusedLocal
def make_doom_env_impl(
        doom_spec,
        cfg=None,
        env_config=None,
        skip_frames=None,
        episode_horizon=None,
        player_id=None, num_agents=None, max_num_players=None, num_bots=0,  # for multi-agent
        custom_resolution=None,
        **kwargs,
):
    skip_frames = skip_frames if skip_frames is not None else cfg.env_frameskip

    fps = cfg.fps if 'fps' in cfg else None
    async_mode = fps == 0

    if player_id is None:
        env = VizdoomEnv(
            doom_spec.action_space, doom_spec.env_spec_file, skip_frames=skip_frames, async_mode=async_mode,
        )
    else:
        from envs.doom.multiplayer.doom_multiagent import VizdoomEnvMultiplayer
        env = VizdoomEnvMultiplayer(
            doom_spec.action_space, doom_spec.env_spec_file,
            player_id=player_id, num_agents=num_agents, max_num_players=max_num_players, num_bots=num_bots,
            skip_frames=skip_frames,
            async_mode=async_mode,
        )

    record_to = cfg.record_to if 'record_to' in cfg else None
    if record_to is not None:
        env = RecordingWrapper(env, record_to)

    env = MultiplayerStatsWrapper(env)

    if num_bots > 0:
        bot_difficulty = cfg.initial_bot_difficulty if 'initial_bot_difficulty' in cfg else None
        env = BotDifficultyWrapper(env, bot_difficulty)

    if custom_resolution is None:
        env = SetResolutionWrapper(env, '256x144')  # default (wide aspect ratio)
    else:
        assert custom_resolution in resolutions
        env = SetResolutionWrapper(env, custom_resolution)

    h, w, channels = env.observation_space.shape
    if w != DOOM_W:
        env = ResizeWrapper(env, DOOM_W, DOOM_H, grayscale=False)

    # randomly vary episode duration to somewhat decorrelate the experience
    timeout = doom_spec.default_timeout
    if episode_horizon is not None and episode_horizon > 0:
        timeout = episode_horizon
    env = TimeLimitWrapper(env, limit=timeout, random_variation_steps=0)

    pixel_format = cfg.pixel_format if 'pixel_format' in cfg else 'HWC'
    if pixel_format == 'CHW':
        env = PixelFormatChwWrapper(env)

    if doom_spec.extra_wrappers is not None:
        for wrapper_cls, wrapper_kwargs in doom_spec.extra_wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)

    if doom_spec.reward_scaling != 1.0:
        env = RewardScalingWrapper(env, doom_spec.reward_scaling)

    return env


def make_doom_multiplayer_env(doom_spec, cfg=None, env_config=None, **kwargs):
    skip_frames = cfg.env_frameskip

    if cfg.num_bots < 0:
        num_bots = doom_spec.num_bots

    num_agents = doom_spec.num_agents if cfg.num_agents <= 0 else cfg.num_agents
    max_num_players = num_agents + cfg.num_humans

    is_multiagent = num_agents > 1

    def make_env_func(player_id):
        return make_doom_env_impl(
            doom_spec,
            cfg=cfg,
            player_id=player_id, num_agents=num_agents, max_num_players=max_num_players, num_bots=num_bots,
            skip_frames=1 if is_multiagent else skip_frames,  # multi-agent skipped frames are handled by the wrapper
            **kwargs,
        )

    if is_multiagent:
        # create a wrapper that treats multiple game instances as a single multi-agent environment

        from envs.doom.multiplayer.doom_multiagent_wrapper import VizdoomMultiAgentEnv
        env = VizdoomMultiAgentEnv(
            num_agents=num_agents,
            make_env_func=make_env_func,
            env_config=env_config,
            skip_frames=skip_frames,
        )
    else:
        # if we have only one agent, there's no need for multi-agent wrapper
        from envs.doom.multiplayer.doom_multiagent_wrapper import init_multiplayer_env
        env = init_multiplayer_env(make_env_func, player_id=0, env_config=env_config)

    return env


def make_doom_env(env_name, **kwargs):
    spec = doom_env_by_name(env_name)

    if spec.num_agents > 1 or spec.num_bots > 0:
        # requires multiplayer setup (e.g. at least a host, not a singleplayer game)
        return make_doom_multiplayer_env(spec, **kwargs)
    else:
        return make_doom_env_impl(spec, **kwargs)
