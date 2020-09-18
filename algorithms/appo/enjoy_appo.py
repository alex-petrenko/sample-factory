import datetime
import math
import os
import sys
import time
from collections import deque
from os.path import join

import numpy as np
import torch

from algorithms.appo.actor_worker import transform_dict_observations
from algorithms.appo.learner import LearnerWorker
from algorithms.appo.model import create_actor_critic
from algorithms.appo.model_utils import get_hidden_size
from algorithms.utils.action_distributions import ContinuousActionDistribution
from algorithms.utils.algo_utils import ExperimentStatus
from algorithms.utils.arguments import parse_args, load_from_checkpoint
from algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from envs.create_env import create_env
from utils.utils import log, AttrDict


def enjoy(cfg, max_num_episodes=1000000, max_num_frames=1e9):
    cfg = load_from_checkpoint(cfg)

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1

    if cfg.record_to:
        tstamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        cfg.record_to = join(cfg.record_to, f'{cfg.experiment}', tstamp)
        if not os.path.isdir(cfg.record_to):
            os.makedirs(cfg.record_to)
    else:
        cfg.record_to = None

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
    # env.seed(0)

    is_multiagent = is_multiagent_env(env)
    if not is_multiagent:
        env = MultiAgentWrapper(env)

    if hasattr(env.unwrapped, 'reset_on_init'):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    episode_rewards = []
    true_rewards = deque([], maxlen=100)
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    obs = env.reset()

    with torch.no_grad():
        for _ in range(max_num_episodes):
            done = [False] * len(obs)
            rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)

            episode_reward = 0

            while True:
                obs_torch = AttrDict(transform_dict_observations(obs))
                for key, x in obs_torch.items():
                    obs_torch[key] = torch.from_numpy(x).to(device).float()

                policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)

                # sample actions from the distribution by default
                actions = policy_outputs.actions

                action_distribution = policy_outputs.action_distribution
                if isinstance(action_distribution, ContinuousActionDistribution):
                    if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                        actions = action_distribution.means

                actions = actions.cpu().numpy()

                rnn_states = policy_outputs.rnn_states

                for _ in range(render_action_repeat):
                    if not cfg.no_render:
                        target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
                        current_delay = time.time() - last_render_start
                        time_wait = target_delay - current_delay

                        if time_wait > 0:
                            # log.info('Wait time %.3f', time_wait)
                            time.sleep(time_wait)

                        last_render_start = time.time()
                        env.render()

                    obs, rew, done, infos = env.step(actions)

                    episode_reward += np.mean(rew)
                    num_frames += 1

                    if all(done):
                        true_rewards.append(infos[0].get('true_reward', math.nan))
                        log.info('Episode finished at %d frames', num_frames)
                        if not math.isnan(np.mean(true_rewards)):
                            log.info('true rew %.3f avg true rew %.3f', true_rewards[-1], np.mean(true_rewards))

                        # VizDoom multiplayer stuff
                        # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                        #     key = f'PLAYER{player}_FRAGCOUNT'
                        #     if key in infos[0]:
                        #         log.debug('Score for player %d: %r', player, infos[0][key])
                        break

                if all(done) or max_frames_reached(num_frames):
                    break

            if not cfg.no_render:
                env.render()
            time.sleep(0.01)

            episode_rewards.append(episode_reward)
            last_episodes = episode_rewards[-100:]
            avg_reward = sum(last_episodes) / len(last_episodes)
            log.info(
                'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
            )

            if max_frames_reached(num_frames):
                break

    env.close()

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def main():
    """Script entry point."""
    cfg = parse_args(evaluation=True)
    status, avg_reward = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
