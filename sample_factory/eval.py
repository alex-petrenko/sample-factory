import json
import time
from collections import deque
from pathlib import Path
from typing import Deque

import numpy as np
import pandas as pd
from signal_slot.signal_slot import StatusCode

from sample_factory.algo.sampling.evaluation_sampling_api import EvalSamplingAPI
from sample_factory.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir, log


def _print_fps_stats(cfg: Config, fps_stats: Deque):
    episodes_sampled = fps_stats[-1][1]
    env_steps_sampled = fps_stats[-1][2]
    delta_sampled = env_steps_sampled - fps_stats[0][2]
    delta_time = fps_stats[-1][0] - fps_stats[0][0]
    fps = delta_sampled / delta_time
    fps_frameskip = fps * cfg.env_frameskip
    fps_frameskip_str = f" ({fps_frameskip:.1f} FPS with frameskip)" if cfg.env_frameskip > 1 else ""
    log.info(
        f"Episodes collected: {episodes_sampled}, Samples collected: {env_steps_sampled}, throughput: {fps:.1f} FPS{fps_frameskip_str}"
    )


def _print_eval_summaries(cfg, eval_stats):
    for policy_id in range(cfg.num_policies):
        results = {}
        for key, stat in eval_stats.items():
            stat_value = np.mean(stat[policy_id])

            if "/" in key:
                # custom summaries have their own sections in tensorboard
                avg_tag = key
                min_tag = f"{key}_min"
                max_tag = f"{key}_max"
            elif key in ("reward", "len"):
                # reward and length get special treatment
                avg_tag = f"{key}/{key}"
                min_tag = f"{key}/{key}_min"
                max_tag = f"{key}/{key}_max"
            else:
                avg_tag = f"policy_stats/avg_{key}"
                min_tag = f"policy_stats/avg_{key}_min"
                max_tag = f"policy_stats/avg_{key}_max"

            results[avg_tag] = float(stat_value)

            # for key stats report min/max as well
            if key in ("reward", "true_objective", "len"):
                results[min_tag] = float(min(stat[policy_id]))
                results[max_tag] = float(max(stat[policy_id]))

        log.info(json.dumps(results, indent=4))


def _save_eval_results(cfg, eval_stats):
    for policy_id in range(cfg.num_policies):
        data = {}
        for key, stat in eval_stats.items():
            data[key] = stat[policy_id]

        csv_output_dir = Path(experiment_dir(cfg=cfg))
        if cfg.csv_folder_name is not None:
            csv_output_dir = csv_output_dir / cfg.csv_folder_name
        csv_output_dir.mkdir(exist_ok=True, parents=True)
        csv_output_path = csv_output_dir / f"eval_p{policy_id}.csv"

        data = pd.DataFrame(data)
        data.to_csv(csv_output_path)


def generate_trajectories(cfg: Config, env_info: EnvInfo, sample_env_episodes: int = 1024) -> StatusCode:
    sampler = EvalSamplingAPI(cfg, env_info)
    sampler.init()
    sampler.start()

    print_interval_sec = 1.0
    fps_stats = deque([(time.time(), 0, 0)], maxlen=10)
    episodes_sampled = 0
    last_print = time.time()

    while episodes_sampled < sample_env_episodes:
        try:
            if time.time() - last_print > print_interval_sec:
                # for now we only look at the first policy
                policy_id = 0
                episodes_sampled = len(sampler.eval_episodes[policy_id])
                env_steps_sampled = sampler.total_samples

                fps_stats.append((time.time(), episodes_sampled, env_steps_sampled))
                _print_fps_stats(cfg, fps_stats)
                last_print = time.time()

                log.info(f"Progress: {episodes_sampled}/{sample_env_episodes} episodes sampled")
        except KeyboardInterrupt:
            log.info(f"KeyboardInterrupt in {generate_trajectories.__name__}()")
            break

    status = sampler.stop()

    # TODO: log results to tensorboard?
    _print_eval_summaries(cfg, sampler.eval_stats)
    _save_eval_results(cfg, sampler.eval_stats)

    return status


def do_eval(cfg: Config) -> StatusCode:
    # should always be set to True for this script
    cfg.episode_counter = True
    # decorrelation isn't needed in eval, it only slows us down
    cfg.decorrelate_envs_on_one_worker = False
    env_info = obtain_env_info_in_a_separate_process(cfg)
    return generate_trajectories(cfg, env_info, cfg.sample_env_episodes)
