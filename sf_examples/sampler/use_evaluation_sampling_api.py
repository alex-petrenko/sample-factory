import sys
import time
from collections import deque
from typing import Deque

import numpy as np
from signal_slot.signal_slot import StatusCode

from sample_factory.algo.sampling.evaluation_sampling_api import EvalSamplingAPI
from sample_factory.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_factory.algo.utils.rl_utils import samples_per_trajectory
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log
from sf_examples.atari.train_atari import parse_atari_args, register_atari_components


def _print_fps_stats(cfg: Config, fps_stats: Deque):
    episodes_sampled = fps_stats[-1][1]
    env_steps_sampled = fps_stats[-1][2]
    delta_sampled = env_steps_sampled - fps_stats[0][2]
    delta_time = fps_stats[-1][0] - fps_stats[0][0]
    fps = delta_sampled / delta_time
    fps_frameskip = fps * cfg.env_frameskip
    fps_frameskip_str = f" ({fps_frameskip:.1f} FPS with frameskip)" if cfg.env_frameskip > 1 else ""
    log.debug(
        f"Episodes collected: {episodes_sampled}, Samples collected: {env_steps_sampled}, throughput: {fps:.1f}, FPS{fps_frameskip_str}"
    )


def generate_trajectories(cfg: Config, env_info: EnvInfo, sample_env_episodes: int = 1024) -> StatusCode:
    sampler = EvalSamplingAPI(cfg, env_info)
    sampler.start()

    batch_size = cfg.batch_size // cfg.rollout
    max_episode_number = sample_env_episodes // batch_size

    print_interval_sec = 1.0
    fps_stats = deque([(time.time(), 0, 0)], maxlen=10)
    episodes_sampled = 0
    env_steps_sampled = 0
    last_print = time.time()

    while episodes_sampled < sample_env_episodes:
        try:
            trajectory = sampler.get_trajectories_sync()
            if trajectory is None:
                break

            episode_numbers = sampler.sampling_loop.policy_avg_stats.get(
                "episode_number", [[] for _ in range(cfg.num_policies)]
            )
            # TODO: for now we only look at the first policy, but should handle all later
            episode_numbers = np.array(episode_numbers[0])
            valid = episode_numbers < max_episode_number
            episode_numbers = episode_numbers[valid]

            episodes_sampled = episode_numbers.sum()
            env_steps_sampled += samples_per_trajectory(trajectory)

            if time.time() - last_print > print_interval_sec:
                fps_stats.append((time.time(), episodes_sampled, env_steps_sampled))
                _print_fps_stats(cfg, fps_stats)
                last_print = time.time()
        except KeyboardInterrupt:
            log.info(f"KeyboardInterrupt in {generate_trajectories.__name__}()")
            break

    status = sampler.stop()
    return status


def main() -> StatusCode:
    register_atari_components()
    # TODO: add more arguments
    cfg = parse_atari_args(
        [
            "--env=atari_mspacman",
            "--batch_size=2048",
            "--episode_counter=True",
            # "--serial_mode=True",
        ]
    )
    env_info = obtain_env_info_in_a_separate_process(cfg)

    return generate_trajectories(cfg, env_info)


if __name__ == "__main__":
    sys.exit(main())
