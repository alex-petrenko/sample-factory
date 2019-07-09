from gym.spaces import Discrete
from ray.tune import register_env

from envs.doom.doom_gym import VizdoomEnv
from envs.doom.multiplayer.doom_multiagent import VizdoomEnvMultiplayer, VizdoomMultiAgentEnv, init_multiplayer_env
from envs.doom.wrappers.action_space import doom_action_space
from envs.doom.wrappers.additional_input import DoomAdditionalInputAndRewards
from envs.doom.wrappers.observation_space import SetResolutionWrapper
from envs.doom.wrappers.step_human_input import StepHumanInput
from envs.env_wrappers import ResizeWrapper, RewardScalingWrapper, TimeLimitWrapper

DOOM_W = 128
DOOM_H = 72
DEFAULT_FRAMESKIP = 4


class DoomCfg:
    def __init__(
            self, name, env_cfg, action_space, reward_scaling, default_timeout,
            num_agents=1, num_bots=0,
            no_idle=False,
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


DOOM_ENVS = [
    DoomCfg('doom_basic', 'basic.cfg', Discrete(3), 0.01, 300, no_idle=True),

    DoomCfg('doom_battle', 'D3_battle.cfg', Discrete(9), 1.0, 2100),
    DoomCfg('doom_battle_tuple_actions', 'D3_battle.cfg', doom_action_space(), 1.0, 2100),

    DoomCfg('doom_battle2', 'D4_battle2.cfg', Discrete(9), 1.0, 2100),

    DoomCfg('doom_dm', 'cig.cfg', doom_action_space(), 1.0, int(1e9), num_agents=8),

    DoomCfg('doom_dwango5', 'dwango5_dm.cfg', doom_action_space(), 1.0, int(1e9), num_agents=8),
    DoomCfg('doom_dwango5_bots', 'dwango5_dm.cfg', doom_action_space(), 1.0, int(1e9), num_agents=1, num_bots=7),
]


def doom_env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Doom env')


# noinspection PyUnusedLocal
def make_doom_env(
        doom_cfg, mode='train',
        skip_frames=DEFAULT_FRAMESKIP, human_input=False,
        show_automap=False, episode_horizon=None,
        player_id=None, max_num_players=None, num_bots=0,  # for multi-agent
        env_config=None,
        **kwargs,
):
    skip_frames = skip_frames if skip_frames is not None else 1
    async_mode = False
    if mode == 'test':
        skip_frames = 1
        async_mode = True

    if player_id is None:
        env = VizdoomEnv(doom_cfg.action_space, doom_cfg.env_cfg, skip_frames=skip_frames, async_mode=async_mode)
    else:
        # skip_frames is handled by multi-agent wrapper
        env = VizdoomEnvMultiplayer(
            doom_cfg.action_space, doom_cfg.env_cfg,
            player_id=player_id, max_num_players=max_num_players, num_bots=num_bots,
            skip_frames=skip_frames,
            async_mode=async_mode,
        )

    env.no_idle_action = doom_cfg.no_idle

    if human_input:
        env = StepHumanInput(env)

    # TODO: render higher resolution at test time?
    if mode == 'human':
        env = SetResolutionWrapper(env, '1280x720')
    else:
        env = SetResolutionWrapper(env, '256x144')

    h, w, channels = env.observation_space.shape
    if w != DOOM_W:
        env = ResizeWrapper(env, DOOM_W, DOOM_H, grayscale=False)

    # randomly vary episode duration to somewhat decorrelate the experience
    timeout = doom_cfg.default_timeout - 50
    if episode_horizon is not None and episode_horizon > 0:
        timeout = episode_horizon
    env = TimeLimitWrapper(env, limit=timeout, random_variation_steps=49)

    if doom_cfg.reward_scaling != 1.0:
        env = RewardScalingWrapper(env, doom_cfg.reward_scaling)

    env = DoomAdditionalInputAndRewards(env)
    return env


def make_doom_multiagent_env(
        doom_cfg, mode='train', num_agents=-1, num_bots=-1, num_humans=0,
        skip_frames=DEFAULT_FRAMESKIP, env_config=None,
        **kwargs,
):
    if mode == 'test':
        skip_frames = 1

    if num_bots < 0:
        num_bots = doom_cfg.num_bots

    num_agents = doom_cfg.num_agents if num_agents <= 0 else num_agents
    max_num_players = num_agents + num_humans

    is_multiagent = num_agents > 1

    def make_env_func(player_id):
        return make_doom_env(
            doom_cfg, mode,
            player_id=player_id, max_num_players=max_num_players, num_bots=num_bots,
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


def register_doom_envs_rllib(**kwargs):
    """Register env factories in RLLib system."""
    singleplayer_envs = ['doom_battle', 'doom_battle_tuple_actions']
    for env_name in singleplayer_envs:
        register_env(env_name, lambda config: make_doom_env(doom_env_by_name(env_name), **kwargs))

    multiplayer_envs = ['doom_dm', 'doom_dwango5', 'doom_dwango5_bots']
    for env_name in multiplayer_envs:
        register_env(
            env_name,
            lambda config: make_doom_multiagent_env(doom_env_by_name(env_name), env_config=config, **kwargs),
        )
