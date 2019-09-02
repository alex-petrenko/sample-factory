import sys
import time
from os.path import join

import cv2

from algorithms.utils.arguments import parse_args, get_algo_class, load_from_checkpoint
from envs.create_env import create_env
from utils.utils import log


def enjoy(cfg, max_num_episodes=1000000, max_num_frames=1e9):
    cfg = load_from_checkpoint(cfg)

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

    env = agent.make_env_func(None)
    env.seed(0)

    episode_rewards = []
    num_frames = 0
    done = False

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    for _ in range(max_num_episodes):
        obs = env.reset()
        rnn_states = None
        episode_reward = 0

        while True:
            actions, rnn_states, res = agent.best_action([obs], [done], rnn_states, deterministic=False)

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

                obs, rew, done, _ = env.step(actions[0])
                episode_reward += rew
                num_frames += 1

                agent._update_memory(actions, res.memory_write, [done])

                if done:
                    log.info('Episode finished at %d frames', num_frames)
                    break

            if done or max_frames_reached(num_frames):
                break

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
