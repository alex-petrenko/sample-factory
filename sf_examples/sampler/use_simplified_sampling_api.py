import sys
import time

from signal_slot.signal_slot import StatusCode

from sample_factory.algo.sampling.simplified_sampling_api import SyncSamplingAPI
from sample_factory.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_factory.algo.utils.rl_utils import samples_per_trajectory
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log
from sf_examples.atari_examples.train_atari import parse_atari_args, register_atari_components


def generate_trajectories(cfg: Config, env_info: EnvInfo, sample_env_steps: int = 1_000_000) -> StatusCode:
    sampler = SyncSamplingAPI(cfg, env_info)
    sampler.start()

    print_interval_sec = 1.0
    last_print = time.time()
    prev_sampled = sampled = 0

    while sampled < sample_env_steps:
        try:
            trajectory = sampler.get_trajectories_sync()
            if trajectory is None:
                break

            sampled += samples_per_trajectory(trajectory)

            if time.time() - last_print > print_interval_sec:
                fps = (sampled - prev_sampled) / (time.time() - last_print)
                fps_frameskip = fps * cfg.env_frameskip
                fps_frameskip_str = f" ({fps_frameskip:.1f} FPS with frameskip)" if cfg.env_frameskip > 1 else ""
                log.debug(f"Samples collected: {sampled}, throughput: {fps:.1f} FPS{fps_frameskip_str}")
                last_print = time.time()
                prev_sampled = sampled
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
