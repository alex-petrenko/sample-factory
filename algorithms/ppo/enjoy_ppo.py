import sys
import time
from os.path import join

import cv2

from algorithms.ppo.agent_ppo import AgentPPO
from algorithms.utils.arguments import parse_args
from envs.create_env import create_env
from utils.utils import log


def enjoy(args, params, max_num_episodes=1000000, max_num_frames=1e9):
    params = params.load()

    def make_env_func(env_config):
        # whether to run Doom env at it's default FPS (ASYNC mode)
        async_mode = args.fps == 0

        if args.record_to is not None:
            args.record_to = join(args.record_to, f'{args.env}_{args.experiment}')

        return create_env(
            args.env,
            pixel_format=args.pixel_format,
            async_mode=async_mode, skip_frames=args.evaluation_env_frameskip,
            num_agents=args.num_agents, num_bots=args.num_bots, num_humans=args.num_humans,
            bot_difficulty=args.bot_difficulty,
            record_to=args.record_to,
            env_config=env_config,
        )

    agent = AgentPPO(make_env_func, params=params)
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
            actions, rnn_states = agent.best_action([obs], [done], rnn_states, deterministic=False)

            for _ in range(args.render_action_repeat):
                if not args.no_render:
                    target_delay = 1.0 / args.fps if args.fps > 0 else 0
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
    args, params = parse_args(AgentPPO.Params, evaluation=True)
    return enjoy(args, params)


if __name__ == '__main__':
    sys.exit(main())
