import json
import sys
import time
from os.path import join

import cv2

from algorithms.utils.arguments import parse_args, load_from_checkpoint
from envs.create_env import create_env
from utils.utils import log


# noinspection DuplicatedCode
def enjoy(cfg, max_num_episodes=1000000, max_num_frames=1e9):
    cfg = load_from_checkpoint(cfg)

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    cfg.env_frameskip = 1  # for evaluation

    if cfg.record_to is not None:
        cfg.record_to = join(cfg.record_to, f'{cfg.env}_{cfg.experiment}')

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_env_func(None)
    env.seed(0)

    actions_file = '/home/apetrenk/all/projects/doom/recorded_episodes/doom_battle_hybrid_doom_battle_hybrid_torch_v7_seed2/2019_08_19--14_54_51/ep_000_r47.45/actions.json'
    with open(actions_file, 'r') as actions_fobj:
        actions_list = json.load(actions_fobj)
        actions_i = 0

    episode_rewards = []
    num_frames = 0
    done = False

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    for _ in range(max_num_episodes):
        obs = env.reset()
        episode_reward = 0

        while True:
            actions = actions_list[actions_i]
            actions_i += 1
            # log.info('Action idx: %d, num_actions %d', actions_i, len(actions_list))

            for _ in range(render_action_repeat):
                if not cfg.no_render:
                    target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
                    current_delay = time.time() - last_render_start
                    time_wait = target_delay - current_delay

                    if time_wait > 0:
                        time.sleep(time_wait)

                    last_render_start = time.time()
                    env.render()

                obs, rew, done, _ = env.step(actions)
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

    env.close()
    cv2.destroyAllWindows()


def main():
    """Script entry point."""
    cfg = parse_args(evaluation=True)
    return enjoy(cfg)


if __name__ == '__main__':
    sys.exit(main())
