from gym.spaces import Discrete

from envs.doom.doom_gym import VizdoomEnv
from envs.doom.multiplayer.doom_multiagent import VizdoomEnvMultiplayer, VizdoomMultiAgentEnv, init_multiplayer_env
from envs.doom.action_space import doom_action_space, doom_action_space_no_weap, doom_action_space_discrete, \
    doom_action_space_hybrid, doom_action_space_hybrid_no_weap, doom_action_space_experimental, doom_action_space_basic
from envs.doom.wrappers.additional_input import DoomAdditionalInputAndRewards
from envs.doom.wrappers.bot_difficulty import BotDifficultyWrapper
from envs.doom.wrappers.multiplayer_stats import MultiplayerStatsWrapper
from envs.doom.wrappers.observation_space import SetResolutionWrapper, resolutions
from envs.doom.wrappers.scenario_wrappers.gathering_reward_shaping import DoomGatheringRewardShaping
from envs.doom.wrappers.step_human_input import StepHumanInput
from envs.env_wrappers import ResizeWrapper, RewardScalingWrapper, TimeLimitWrapper, RecordingWrapper, \
    PixelFormatChwWrapper

DOOM_W = 128
DOOM_H = 72


DEFAULT_CONFIG = {
    'skip_frames': 4,
}


def cfg_param(name, cfg=None):
    value = None
    if cfg is not None:
        value = cfg.get(name, None)

    if value is None:
        value = DEFAULT_CONFIG[name]

    return value


class DoomCfg:
    def __init__(
            self, name, env_cfg, action_space, reward_scaling=1.0, default_timeout=int(1e9),
            num_agents=1, num_bots=0,
            no_idle=False,
            extra_wrappers=None,
    ):
        self.name = name
        self.env_cfg = env_cfg
        self.action_space = action_space
        self.reward_scaling = reward_scaling
        self.default_timeout = default_timeout

        # set to True if the environment does not assume an IDLE action
        self.no_idle = no_idle

        # 1 for singleplayer, >1 otherwise
        self.num_agents = num_agents

        # CLI arguments override this (see enjoy.py)
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
    DoomCfg(
        'doom_basic', 'basic.cfg',
        Discrete(4),  # idle, left, right, attack
        0.01, 300,
        extra_wrappers=[(DoomAdditionalInputAndRewards, {'with_reward_shaping': False})],
    ),

    DoomCfg('doom_battle_discrete', 'D3_battle.cfg', Discrete(8), 1.0, 2100),
    DoomCfg('doom_battle_tuple_actions', 'D3_battle.cfg', doom_action_space_discrete(), 1.0, 2100),
    DoomCfg('doom_battle_continuous', 'D3_battle_continuous.cfg', doom_action_space_no_weap(), 1.0, 2100),
    DoomCfg('doom_battle_hybrid', 'D3_battle_continuous.cfg', doom_action_space_hybrid_no_weap(), 1.0, 2100),

    DoomCfg('doom_dm', 'cig.cfg', doom_action_space(), 1.0, int(1e9), num_agents=8),

    DoomCfg(
        'doom_two_colors_easy', 'two_colors_easy.cfg',
        Discrete(5),  # idle, left, right, forward, backward
        extra_wrappers=[
            (DoomAdditionalInputAndRewards, {'with_reward_shaping': False}),
            (DoomGatheringRewardShaping, {}),
        ]
    ),

    DoomCfg(
        'doom_two_colors_hard', 'two_colors_hard.cfg',
        Discrete(5),  # idle, left, right, forward, backward
        extra_wrappers=[
            (DoomAdditionalInputAndRewards, {'with_reward_shaping': False}),
            (DoomGatheringRewardShaping, {}),
        ]
    ),

    DoomCfg('doom_dwango5', 'dwango5_dm.cfg', doom_action_space(), 1.0, int(1e9), num_agents=8),

    DoomCfg(
        'doom_dwango5_bots',
        'dwango5_dm.cfg',
        doom_action_space_discrete(),
        1.0, int(1e9),
        num_agents=1, num_bots=7,
    ),

    DoomCfg(
        'doom_dwango5_bots_continuous',
        'dwango5_dm_continuous.cfg',
        doom_action_space(),
        1.0, int(1e9),
        num_agents=1, num_bots=7,
    ),

    DoomCfg(
        'doom_dwango5_bots_hybrid',
        'dwango5_dm_continuous.cfg',
        doom_action_space_hybrid(),
        1.0, int(1e9),
        num_agents=1, num_bots=7,
    ),

    DoomCfg(
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
        doom_cfg,
        skip_frames=None,  # non-None value overrides the env_config
        human_input=False,
        show_automap=False, episode_horizon=None,
        player_id=None, num_agents=None, max_num_players=None, num_bots=0,  # for multi-agent
        bot_difficulty=None,
        env_config=None,
        async_mode=False,
        record_to=None,
        custom_resolution=None,
        pixel_format='HWC',
        **kwargs,
):
    env_config = DEFAULT_CONFIG if env_config is None else env_config

    skip_frames = skip_frames if skip_frames is not None else cfg_param('skip_frames', env_config)

    if player_id is None:
        env = VizdoomEnv(
            doom_cfg.action_space, doom_cfg.env_cfg, skip_frames=skip_frames, async_mode=async_mode,
            no_idle_action=doom_cfg.no_idle,
        )
    else:
        # skip_frames is handled by multi-agent wrapper
        env = VizdoomEnvMultiplayer(
            doom_cfg.action_space, doom_cfg.env_cfg,
            player_id=player_id, num_agents=num_agents, max_num_players=max_num_players, num_bots=num_bots,
            skip_frames=skip_frames,
            async_mode=async_mode,
            no_idle_action=doom_cfg.no_idle,
        )

    if doom_cfg.reward_scaling != 1.0:
        env = RewardScalingWrapper(env, doom_cfg.reward_scaling)

    if record_to is not None:
        env = RecordingWrapper(env, record_to)

    env = MultiplayerStatsWrapper(env)
    if num_bots > 0:
        env = BotDifficultyWrapper(env, bot_difficulty)

    if human_input:
        env = StepHumanInput(env)

    if custom_resolution is None:
        env = SetResolutionWrapper(env, '256x144')  # default (wide aspect ratio)
    else:
        assert custom_resolution in resolutions
        env = SetResolutionWrapper(env, custom_resolution)

    h, w, channels = env.observation_space.shape
    if w != DOOM_W:
        env = ResizeWrapper(env, DOOM_W, DOOM_H, grayscale=False)

    # randomly vary episode duration to somewhat decorrelate the experience
    timeout = doom_cfg.default_timeout - 50
    if episode_horizon is not None and episode_horizon > 0:
        timeout = episode_horizon
    env = TimeLimitWrapper(env, limit=timeout, random_variation_steps=49)

    if pixel_format == 'CHW':
        env = PixelFormatChwWrapper(env)

    if doom_cfg.extra_wrappers is not None:
        for wrapper_cls, wrapper_kwargs in doom_cfg.extra_wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)

    return env


def make_doom_multiplayer_env(
        doom_cfg, num_agents=-1, num_bots=-1, num_humans=0,
        skip_frames=None, env_config=None,
        **kwargs,
):
    env_config = DEFAULT_CONFIG if env_config is None else env_config
    skip_frames = skip_frames if skip_frames is not None else cfg_param('skip_frames', env_config)

    if num_bots < 0:
        num_bots = doom_cfg.num_bots

    num_agents = doom_cfg.num_agents if num_agents <= 0 else num_agents
    max_num_players = num_agents + num_humans

    is_multiagent = num_agents > 1

    def make_env_func(player_id):
        return make_doom_env_impl(
            doom_cfg,
            player_id=player_id, num_agents=num_agents, max_num_players=max_num_players, num_bots=num_bots,
            skip_frames=1 if is_multiagent else skip_frames,  # multi-agent skipped frames are handled by the wrapper
            **kwargs,
        )

    if is_multiagent:
        env = VizdoomMultiAgentEnv(
            num_agents=num_agents,
            make_env_func=make_env_func,
            env_config=env_config,
            skip_frames=skip_frames,
        )
    else:
        # if we have only one agent, there's no need for multi-agent wrapper
        env = init_multiplayer_env(make_env_func, player_id=0, env_config=env_config)

    return env


def make_doom_env(env_name, **kwargs):
    cfg = doom_env_by_name(env_name)

    if cfg.num_agents > 1 or cfg.num_bots > 0:
        # requires multiplayer setup (e.g. at least a host, not a singleplayer game)
        return make_doom_multiplayer_env(cfg, **kwargs)
    else:
        return make_doom_env_impl(cfg, **kwargs)
