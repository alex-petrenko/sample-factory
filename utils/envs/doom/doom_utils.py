import gym
# noinspection PyUnresolvedReferences
import vizdoomgym

from utils.envs.doom.multiplayer.doom_multiagent import VizdoomEnvMultiplayer
from utils.envs.doom.wrappers.additional_input import DoomAdditionalInput
from utils.envs.doom.wrappers.observation_space import SetResolutionWrapper
from utils.envs.doom.wrappers.step_human_input import StepHumanInput
from utils.envs.env_wrappers import ResizeWrapper, RewardScalingWrapper, TimeLimitWrapper

DOOM_W = 128
DOOM_H = 72


class DoomCfg:
    def __init__(self, name, env_id, reward_scaling, default_timeout):
        self.name = name
        self.env_id = env_id
        self.reward_scaling = reward_scaling
        self.default_timeout = default_timeout


DOOM_ENVS = [
    DoomCfg('doom_basic', 'VizdoomBasic-v0', 0.01, 300),
    DoomCfg('doom_maze', 'VizdoomMyWayHome-v0', 1.0, 2100),
    DoomCfg('doom_maze_sparse', 'VizdoomMyWayHomeSparse-v0', 1.0, 2100),
    DoomCfg('doom_maze_very_sparse', 'VizdoomMyWayHomeVerySparse-v0', 1.0, 2100),
    DoomCfg('doom_maze_multi_goal', 'VizdoomMyWayHomeMultiGoal-v0', 1.0, 2100),
    DoomCfg('doom_maze_multi_goal_random', 'VizdoomMyWayHomeMultiGoalRandom-v0', 1.0, 2100),
    DoomCfg('doom_maze_no_goal', 'VizdoomMyWayHomeNoGoal-v0', 1.0, 20000),
    DoomCfg('doom_maze_no_goal_random', 'VizdoomMyWayHomeNoGoalRandom-v0', 1.0, 20000),

    DoomCfg('doom_maze_goal', 'VizdoomMyWayHomeGoal-v0', 1.0, 2100),

    DoomCfg('doom_maze_sptm', 'VizdoomSptmBattleNavigation-v0', 1.0, 2100),

    DoomCfg('doom_textured_easy', 'VizdoomTexturedMazeEasy-v0', 1.0, 20000),
    DoomCfg('doom_textured_very_sparse', 'VizdoomTexturedMazeVerySparse-v0', 1.0, 20000),
    DoomCfg('doom_textured', 'VizdoomTexturedMaze-v0', 1.0, 2100),

    DoomCfg('doom_textured_super_sparse', 'VizdoomTexturedMazeSuperSparse-v0', 1.0, 20000),
    DoomCfg('doom_textured_super_sparse_v2', 'VizdoomTexturedMazeSuperSparse-v2', 1.0, 1e9),
    DoomCfg('doom_textured_multi_goal', 'VizdoomTexturedMazeMultiGoal-v0', 1.0, 20000),

    DoomCfg('doom_textured_large_no_goal', 'VizdoomTexturedMazeLargeNoGoal-v0', 1.0, 2100),
    DoomCfg('doom_textured_no_goal_random', 'VizdoomTexturedMazeNoGoalRandom-v0', 1.0, 20000),
    DoomCfg('doom_textured_random_goal_v2', 'VizdoomTexturedMazeRandomGoal-v2', 1.0, 20000),

    DoomCfg('doom_battle', 'VizdoomBattle-v0', 1.0, 2100),
    DoomCfg('doom_battle2', 'VizdoomBattle2-v0', 1.0, 2100),

    DoomCfg('doom_dm', 29, 1.0, int(1e9)),
    DoomCfg('doom_dm_test', 30, 1.0, int(1e9)),
]


def doom_env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Doom env')


# noinspection PyUnusedLocal
def make_doom_env(
        doom_cfg, mode='train',
        skip_frames=True, human_input=False,
        show_automap=False, episode_horizon=None,
        player_id=None, num_players=None,  # for multi-agent
        **kwargs,
):
    skip_frames = 4 if skip_frames else 1

    if player_id is None:
        env = gym.make(doom_cfg.env_id, show_automap=show_automap, skip_frames=skip_frames)
    else:
        env = VizdoomEnvMultiplayer(
            doom_cfg.env_id, player_id=player_id, num_players=num_players, skip_frames=skip_frames,
        )

    if human_input:
        env = StepHumanInput(env)

    # courtesy of https://github.com/pathak22/noreward-rl/blob/master/src/envs.py
    # and https://github.com/ppaquette/gym-doom
    if mode == 'test':
        env = SetResolutionWrapper(env, '800x450')
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

    env = DoomAdditionalInput(env)
    return env
