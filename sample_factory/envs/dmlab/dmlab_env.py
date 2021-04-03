import os
from collections import deque

import numpy as np

from sample_factory.algorithms.utils.algo_utils import EXTRA_EPISODIC_STATS_PROCESSING, EXTRA_PER_POLICY_SUMMARIES
from sample_factory.envs.dmlab.dmlab30 import dmlab30_level_name_to_level, \
    DMLAB30_LEVELS_THAT_USE_LEVEL_CACHE, DMLAB30_LEVELS, HUMAN_SCORES, RANDOM_SCORES, LEVEL_MAPPING
from sample_factory.envs.dmlab.dmlab_gym import DmlabGymEnv, dmlab_level_to_level_name
from sample_factory.envs.dmlab.dmlab_level_cache import dmlab_ensure_global_cache_initialized
from sample_factory.envs.dmlab.dmlab_model import dmlab_register_models
from sample_factory.envs.dmlab.wrappers.reward_shaping import DmlabRewardShapingWrapper, RAW_SCORE_SUMMARY_KEY_SUFFIX
from sample_factory.envs.env_wrappers import PixelFormatChwWrapper, RecordingWrapper
from sample_factory.utils.utils import log, experiment_dir, static_vars

DMLAB_INITIALIZED = False


def get_dataset_path(cfg):
    cfg_dataset_path = os.path.expanduser(cfg.dmlab30_dataset)
    return cfg_dataset_path


class DmLabSpec:
    def __init__(self, name, level, extra_cfg=None):
        self.name = name
        self.level = level
        self.extra_cfg = {} if extra_cfg is None else extra_cfg


DMLAB_ENVS = [
    DmLabSpec('dmlab_benchmark', 'contributed/dmlab30/rooms_collect_good_objects_train'),

    # train a single agent for all 30 DMLab tasks
    DmLabSpec('dmlab_30', [dmlab30_level_name_to_level(l) for l in DMLAB30_LEVELS]),
    DmLabSpec('dmlab_level_cache', [dmlab30_level_name_to_level(l) for l in DMLAB30_LEVELS_THAT_USE_LEVEL_CACHE]),

    # this is very hard to work with as a benchmark, because FPS fluctuates a lot due to slow resets.
    # also depends a lot on whether levels are in level cache or not
    DmLabSpec('dmlab_benchmark_slow_reset', 'contributed/dmlab30/rooms_keys_doors_puzzle'),

    DmLabSpec('dmlab_sparse', 'contributed/dmlab30/explore_goal_locations_large'),
    DmLabSpec('dmlab_very_sparse', 'contributed/dmlab30/explore_goal_locations_large', extra_cfg={'minGoalDistance': '10'}),
    DmLabSpec('dmlab_sparse_doors', 'contributed/dmlab30/explore_obstructed_goals_large'),
    DmLabSpec('dmlab_nonmatch', 'contributed/dmlab30/rooms_select_nonmatching_object'),
    DmLabSpec('dmlab_watermaze', 'contributed/dmlab30/rooms_watermaze'),
]


def dmlab_env_by_name(name):
    for spec in DMLAB_ENVS:
        if spec.name == name:
            return spec

    # not a known "named" environment with a predefined spec
    log.warning('Level %s not found. Interpreting the level name as an unmodified DMLab-30 env name!', name)
    level = name.split('dmlab_')[1]
    spec = DmLabSpec(name, level)
    return spec


def get_task_id(env_config, spec, cfg):
    if env_config is None:
        return 0
    elif isinstance(spec.level, str):
        return 0
    elif isinstance(spec.level, (list, tuple)):
        num_envs = len(spec.level)

        if cfg.dmlab_one_task_per_worker:
            return env_config['worker_index'] % num_envs
        else:
            return env_config['env_id'] % num_envs
    else:
        raise Exception('spec level is either string or a list/tuple')


def task_id_to_level(task_id, spec):
    if isinstance(spec.level, str):
        return spec.level
    elif isinstance(spec.level, (list, tuple)):
        levels = spec.level
        level = levels[task_id]
        return level
    else:
        raise Exception('spec level is either string or a list/tuple')


def list_all_levels_for_experiment(env_name):
    spec = dmlab_env_by_name(env_name)
    if isinstance(spec.level, str):
        return [spec.level]
    elif isinstance(spec.level, (list, tuple)):
        levels = spec.level
        return levels
    else:
        raise Exception('spec level is either string or a list/tuple')


# noinspection PyUnusedLocal
def make_dmlab_env_impl(spec, cfg, env_config, **kwargs):
    skip_frames = cfg.env_frameskip

    gpu_idx = 0
    if len(cfg.dmlab_gpus) > 0:
        if kwargs.get('env_config') is not None:
            vector_index = kwargs['env_config']['vector_index']
            gpu_idx = cfg.dmlab_gpus[vector_index % len(cfg.dmlab_gpus)]
            log.debug('Using GPU %d for DMLab rendering!', gpu_idx)

    task_id = get_task_id(env_config, spec, cfg)
    level = task_id_to_level(task_id, spec)
    log.debug('%r level %s task id %d', env_config, level, task_id)

    env = DmlabGymEnv(
        task_id, level, skip_frames, cfg.res_w, cfg.res_h, cfg.dmlab_throughput_benchmark, cfg.dmlab_renderer,
        get_dataset_path(cfg), cfg.dmlab_with_instructions, cfg.dmlab_extended_action_set,
        cfg.dmlab_use_level_cache, cfg.dmlab_level_cache_path,
        gpu_idx, spec.extra_cfg,
    )

    if env_config and 'env_id' in env_config:
        env.seed(env_config['env_id'])

    if 'record_to' in cfg and cfg.record_to is not None:
        env = RecordingWrapper(env, cfg.record_to, 0)

    if cfg.pixel_format == 'CHW':
        env = PixelFormatChwWrapper(env)

    env = DmlabRewardShapingWrapper(env)
    return env


def make_dmlab_env(env_name, cfg=None, **kwargs):
    ensure_initialized(cfg, env_name)

    spec = dmlab_env_by_name(env_name)
    return make_dmlab_env_impl(spec, cfg=cfg, **kwargs)


@static_vars(new_level_returns=dict(), env_spec=None)
def dmlab_extra_episodic_stats_processing(policy_id, stat_key, stat_value, cfg):
    if RAW_SCORE_SUMMARY_KEY_SUFFIX not in stat_key:
        return

    new_level_returns = dmlab_extra_episodic_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        new_level_returns[policy_id] = dict()

    if dmlab_extra_episodic_stats_processing.env_spec is None:
        dmlab_extra_episodic_stats_processing.env_spec = dmlab_env_by_name(cfg.env)

    task_id = int(stat_key.split('_')[1])  # this is a bit hacky but should do the job
    level = task_id_to_level(task_id, dmlab_extra_episodic_stats_processing.env_spec)
    level_name = dmlab_level_to_level_name(level)

    if level_name not in new_level_returns[policy_id]:
        new_level_returns[policy_id][level_name] = []

    new_level_returns[policy_id][level_name].append(stat_value)


@static_vars(all_levels=None)
def dmlab_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    """
    We precisely follow IMPALA repo (scalable_agent) here for the reward calculation.

    The procedure is:
    1. Calculate mean raw episode score for the last few episodes for each level
    2. Calculate human-normalized score using this mean value
    3. Calculate capped score

    The key point is that human-normalization and capping is done AFTER mean, which can lead to slighly higher capped
    scores for levels that exceed the human baseline.

    Another important point: we write the avg score summary only when we have at least one episode result for every
    level. Again, we try to precisely follow IMPALA implementation here.

    """

    new_level_returns = dmlab_extra_episodic_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        return

    # exit if we don't have at least one episode for all levels
    if dmlab_extra_summaries.all_levels is None:
        dmlab_levels = list_all_levels_for_experiment(cfg.env)
        level_names = [dmlab_level_to_level_name(l) for l in dmlab_levels]
        dmlab_extra_summaries.all_levels = level_names

    all_levels = dmlab_extra_summaries.all_levels
    for level in all_levels:
        if len(new_level_returns[policy_id].get(level, [])) < 1:
            return

    level_mean_scores_normalized = []
    level_mean_scores_normalized_capped = []
    for level_idx, level in enumerate(all_levels):
        level_score = new_level_returns[policy_id][level]
        assert len(level_score) > 0

        score = np.mean(level_score)
        test_level_name = LEVEL_MAPPING[level]
        human = HUMAN_SCORES[test_level_name]
        random = RANDOM_SCORES[test_level_name]

        human_normalized_score = (score - random) / (human - random) * 100
        capped_human_normalized_score = min(100.0, human_normalized_score)

        level_mean_scores_normalized.append(human_normalized_score)
        level_mean_scores_normalized_capped.append(capped_human_normalized_score)

        level_key = f'{level_idx:02d}_{level}'
        summary_writer.add_scalar(f'_dmlab/{level_key}_human_norm_score', human_normalized_score, env_steps)
        summary_writer.add_scalar(f'_dmlab/capped_{level_key}_human_norm_score', capped_human_normalized_score, env_steps)

    assert len(level_mean_scores_normalized) == len(level_mean_scores_normalized_capped) == len(all_levels)

    mean_normalized_score = np.mean(level_mean_scores_normalized)
    capped_mean_normalized_score = np.mean(level_mean_scores_normalized_capped)

    # use 000 here to put these summaries on top in tensorboard (it sorts by ASCII)
    summary_writer.add_scalar(f'_dmlab/000_mean_human_norm_score', mean_normalized_score, env_steps)
    summary_writer.add_scalar(f'_dmlab/000_capped_mean_human_norm_score', capped_mean_normalized_score, env_steps)

    # clear the scores and start anew (this is exactly what IMPALA does)
    dmlab_extra_episodic_stats_processing.new_level_returns[policy_id] = dict()

    # add a new stat that PBT can track
    target_objective_stat = 'dmlab_target_objective'
    if target_objective_stat not in policy_avg_stats:
        policy_avg_stats[target_objective_stat] = [deque(maxlen=1) for _ in range(cfg.num_policies)]

    policy_avg_stats[target_objective_stat][policy_id].append(capped_mean_normalized_score)


def ensure_initialized(cfg, env_name):
    global DMLAB_INITIALIZED
    if DMLAB_INITIALIZED:
        return

    dmlab_register_models()

    if env_name == 'dmlab_30':
        # extra functions to calculate human-normalized score etc.
        EXTRA_EPISODIC_STATS_PROCESSING.append(dmlab_extra_episodic_stats_processing)
        EXTRA_PER_POLICY_SUMMARIES.append(dmlab_extra_summaries)

    num_policies = cfg.num_policies if hasattr(cfg, 'num_policies') else 1
    all_levels = list_all_levels_for_experiment(env_name)
    level_cache_dir = cfg.dmlab_level_cache_path
    dmlab_ensure_global_cache_initialized(experiment_dir(cfg=cfg), all_levels, num_policies, level_cache_dir)

    DMLAB_INITIALIZED = True
