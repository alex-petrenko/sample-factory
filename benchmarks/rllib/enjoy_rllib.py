import collections
import os
import pickle
import time
from os.path import join

import gym
import numpy as np
import ray
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.registry import get_agent_class
# noinspection PyProtectedMember
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.sampler import _unbatch_tuple_actions
from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import TupleActions
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.rollout import create_parser, DefaultMapping
from ray.tune.util import merge_dicts

from benchmarks.rllib.vizdoom_model import VizdoomVisionNetwork
from envs.ray_envs import register_doom_envs_rllib
from utils.utils import log


def create_parser_custom():
    parser = create_parser()
    parser.add_argument(
        '--dbg',
        action='store_true',
        help='Full debug mode (also enables local-mode)',
    )
    parser.add_argument(
        '--num-episodes',
        default=1000,
        type=int,
        help='Number of full episodes to rollout',
    )
    parser.add_argument(
        '--num-agents',
        default=-1,
        type=int,
        help='Allows to set number of agents less than number of players, to allow humans to join the match'
             'Default value (-1) means number of agents is the same as max number of players',
    )
    parser.add_argument(
        '--num-bots',
        default=-1,
        type=int,
        help='Add classic (non-neural) bots to the match. If default (-1) then use number of bots specified in env cfg',
    )
    parser.add_argument(
        '--num-humans',
        default=0,
        type=int,
        help='Meatbags want to play?',
    )
    parser.add_argument(
        '--fps',
        default=0,
        type=int,
        help='Enable sync mode with adjustable FPS.'
             'Default (0) means default Doom FPS (~35). Leave at 0 for multiplayer',
    )
    parser.add_argument(
        '--bot-difficulty',
        default=150,
        type=int,
        help='Adjust bot difficulty',
    )
    parser.add_argument(
        '--env-frameskip',
        default=1,
        type=int,
        help='Usually frameskip is handled by the rollout loop for smooth rendering, but this is also an option',
    )
    parser.add_argument(
        '--render-action-repeat',
        default=-1,
        type=int,
        help='Repeat an action that many frames during rollout. -1 (default) means read from env config',
    )
    parser.add_argument(
        '--record-to',
        default=join(os.getcwd(), '..', 'doom_episodes'),
        type=str,
        help='Record episodes to this folder using Doom functionality',
    )
    parser.add_argument(
        '--custom-res',
        default=None,
        type=str,
        help='Custom resolution string (e.g. 1920x1080).'
             'Can affect performance of the model if does not match the resolution the model was trained on!',
    )
    return parser


def run(args, config):
    local_mode = False
    if args.dbg:
        local_mode = True

    ray.init(local_mode=local_mode)

    cls = get_agent_class(args._run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(1e9)

    render_frameskip = args.render_action_repeat
    if render_frameskip == -1:
        # default - read from config
        # fallback to default if env config does not have it
        render_frameskip = cfg_param('skip_frames', config.get('env_config', None))

    log.info('Using render frameskip %d! \n\n\n', render_frameskip)

    rollout_loop(
        agent,
        args.env,
        num_steps, num_episodes=args.num_episodes,
        no_render=args.no_render, fps=args.fps, frameskip=render_frameskip,
    )


# noinspection PyUnusedLocal
def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def rollout_loop(agent, env_name, num_steps, num_episodes, no_render=True, fps=1000, frameskip=1):
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    steps = 0
    full_episodes = 0
    last_render_start = time.time()
    avg_reward = collections.deque([], maxlen=100)

    while steps < (num_steps or steps + 1) and full_episodes < num_episodes:
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id_: state_init[mapping_cache[agent_id_]])
        prev_actions = DefaultMapping(
            lambda agent_id_: action_init[mapping_cache[agent_id_]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_episode = 0.0

        while not done and steps < (num_steps or steps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)

                    if isinstance(env.action_space, gym.spaces.Tuple):
                        a_action = TupleActions(a_action)
                        a_action = _unbatch_tuple_actions(a_action)[0]

                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            rewards = None

            for frame in range(frameskip):
                next_obs, reward, done, _ = env.step(action)
                if done:
                    log.info('Done at steps %d', steps)
                    break

                if rewards is None:
                    rewards = reward

                else:
                    if multiagent:
                        for agent_id, r in reward.items():
                            rewards[agent_id] += r
                    else:
                        rewards += reward

                if not no_render:
                    target_delay = 1.0 / fps if fps > 0 else 0
                    current_delay = time.time() - last_render_start
                    time_wait = target_delay - current_delay

                    # note: ASYNC_PLAYER mode actually makes this sleep redundant
                    if time_wait > 0:
                        # log.info('Wait time %.3f', time_wait)
                        time.sleep(time_wait)

                    last_render_start = time.time()
                    env.render()

                steps += 1
                obs = next_obs

            if multiagent:
                for agent_id, r in rewards.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = rewards

            if multiagent:
                done = done['__all__']
                reward_episode += 0 if rewards is None else sum(rewards.values())
            else:
                reward_episode += 0 if rewards is None else rewards

        full_episodes += 1

        avg_reward.append(reward_episode)
        log.info('Reward episode: %.3f, avg_reward %.3f', reward_episode, np.mean(avg_reward))

    env.reset()  # this guarantees that recordings are saved to disk


def main():
    parser = create_parser_custom()
    args = parser.parse_args()

    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    config['num_workers'] = 0
    config['num_gpus'] = 0
    config['num_envs_per_worker'] = 1

    # whether to run Doom env at it's default FPS (ASYNC mode)
    async_mode = args.fps == 0

    skip_frames = args.env_frameskip

    bot_difficulty = args.bot_difficulty

    record_to = join(args.record_to, f'{config["env"]}_{args._run}')

    custom_resolution = args.custom_res

    register_doom_envs_rllib(
        async_mode=async_mode, skip_frames=skip_frames,
        num_agents=args.num_agents, num_bots=args.num_bots, num_humans=args.num_humans,
        bot_difficulty=bot_difficulty,
        record_to=record_to,
        custom_resolution=custom_resolution,
    )

    ModelCatalog.register_custom_model('vizdoom_vision_model', VizdoomVisionNetwork)

    run(args, config)


if __name__ == '__main__':
    main()
