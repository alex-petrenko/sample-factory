import sys
import time
from os.path import join

import cv2
import numpy as np

from algorithms.utils.arguments import parse_args, get_algo_class, load_from_checkpoint
from algorithms.utils.multi_agent import MultiAgentWrapper
from envs.create_env import create_env
from utils.utils import log, AttrDict


def enjoy(cfg, max_num_episodes=1000000, max_num_frames=1e9):
    # allow to override multiplayer settings
    override_multiplayer_settings = False

    num_agents = num_bots = num_humans = -1
    if override_multiplayer_settings:
        num_agents = cfg.num_agents
        num_bots = cfg.num_bots
        num_humans = cfg.num_humans

    cfg = load_from_checkpoint(cfg)

    if override_multiplayer_settings:
        cfg.num_agents = num_agents
        cfg.num_bots = num_bots
        cfg.num_humans = num_humans

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1

    if cfg.record_to is not None:
        cfg.record_to = join(cfg.record_to, f'{cfg.env}_{cfg.experiment}')

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    agent = get_algo_class(cfg.algo)(make_env_func, cfg)
    agent.initialize()

    env = agent.make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
    env.seed(0)

    is_multiagent = hasattr(env, 'num_agents') and env.num_agents > 1
    if not is_multiagent:
        env = MultiAgentWrapper(env)

    episode_rewards = []
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    for _ in range(max_num_episodes):
        obs = env.reset()

        done = [False] * len(obs)

        rnn_states = None
        episode_reward = 0

        while True:
            actions, rnn_states, res = agent.best_action(obs, done, rnn_states, deterministic=False)

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

                obs, rew, done, _ = env.step(actions)

                episode_reward += np.mean(rew)
                num_frames += 1

                agent._update_memory(actions, res.memory_write, done)

                if all(done):
                    log.info('Episode finished at %d frames', num_frames)
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

    agent.finalize()
    env.close()
    cv2.destroyAllWindows()


def main():
    """Script entry point."""
    cfg = parse_args(evaluation=True)
    return enjoy(cfg)


if __name__ == '__main__':
    sys.exit(main())
