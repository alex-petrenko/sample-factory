import argparse
import os
import shutil
import sys
from os.path import join

import cv2

from sample_factory.cfg.arguments import default_cfg
from sample_factory.utils.utils import log
from sf_examples.vizdoom.doom.doom_utils import doom_env_by_name, make_doom_env, make_doom_env_impl


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", type=str, default=None, required=True)
    parser.add_argument("--demo_path", type=str, default=None, required=True)
    args = parser.parse_args()

    spec = doom_env_by_name(args.env)
    cfg = default_cfg(env=args.env)
    if spec.num_agents <= 1:
        env = make_doom_env(args.env, cfg=cfg, env_config=None, render_mode="rgb_array", custom_resolution="1280x720")
    else:
        env = make_doom_env_impl(
            spec,
            cfg=cfg,
            custom_resolution="1280x720",
            player_id=0,
            num_agents=spec.num_agents,
            max_num_players=spec.num_agents,
            num_bots=spec.num_bots,
            render_mode="rgb_array",
        )

    mode = "replay"
    env.unwrapped.mode = mode
    env.unwrapped.initialize()
    game = env.unwrapped.game

    game.replay_episode(args.demo_path)

    frames_dir = args.demo_path + "_frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    frame_id = 0
    while not game.is_episode_finished():
        # Use advance_action instead of make_action.
        game.advance_action()
        img = env.render()

        frame_name = f"{frame_id:05d}.png"
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is not None:
            cv2.imwrite(join(frames_dir, frame_name), img)

        frame_id += 1

        r = game.get_last_reward()
        log.debug("Reward %.3f at frame %d", r, frame_id)

    game.close()


if __name__ == "__main__":
    sys.exit(main())
