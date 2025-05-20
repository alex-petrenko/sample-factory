import sys
import time
from collections import deque
from typing import Deque

from signal_slot.signal_slot import StatusCode

from sample_factory.algo.sampling.simplified_sampling_api import SyncSamplingAPI
from sample_factory.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_factory.algo.utils.rl_utils import samples_per_trajectory
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log
from sf_examples.atari.train_atari import parse_atari_args, register_atari_components


def _print_fps_stats(cfg: Config, fps_stats: Deque):
    sampled = fps_stats[-1][1]
    delta_sampled = sampled - fps_stats[0][1]
    delta_time = fps_stats[-1][0] - fps_stats[0][0]
    fps = delta_sampled / delta_time
    fps_frameskip = fps * cfg.env_frameskip
    fps_frameskip_str = f" ({fps_frameskip:.1f} FPS with frameskip)" if cfg.env_frameskip > 1 else ""
    log.debug(f"Samples collected: {sampled}, throughput: {fps:.1f} FPS{fps_frameskip_str}")


def generate_trajectories(cfg: Config, env_info: EnvInfo, sample_env_steps: int = 1_000_000) -> StatusCode:
    sampler = SyncSamplingAPI(cfg, env_info)
    sampler.start()

    print_interval_sec = 1.0
    fps_stats = deque([(time.time(), 0)], maxlen=10)
    sampled = 0
    last_print = time.time()

    while sampled < sample_env_steps:
        try:
            trajectory = sampler.get_trajectories_sync()
            if trajectory is None:
                break

            sampled += samples_per_trajectory(trajectory)

            if time.time() - last_print > print_interval_sec:
                fps_stats.append((time.time(), sampled))
                _print_fps_stats(cfg, fps_stats)
                last_print = time.time()
        except KeyboardInterrupt:
            log.info(f"KeyboardInterrupt in {generate_trajectories.__name__}()")
            break

    status = sampler.stop()
    return status


def main() -> StatusCode:
    register_atari_components()
    cfg = parse_atari_args()
    env_info = obtain_env_info_in_a_separate_process(cfg)
    return generate_trajectories(cfg, env_info)


if __name__ == "__main__":
    sys.exit(main())
