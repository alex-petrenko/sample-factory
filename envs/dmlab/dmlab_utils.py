import os

from envs.dmlab.dmlab30 import LEVEL_MAPPING, dmlab30_num_envs, level_name_to_level
from envs.dmlab.dmlab_gym import DmlabGymEnv
from envs.dmlab.dmlab_level_cache import dmlab_ensure_global_cache_initialized
from envs.dmlab.dmlab_model import dmlab_register_models
from envs.dmlab.wrappers.reward_shaping import DmlabRewardShapingWrapper
from envs.env_wrappers import PixelFormatChwWrapper, RecordingWrapper
from utils.utils import log, experiment_dir

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
    DmLabSpec('dmlab_30', None),

    # this is very hard to work with as a benchmark, because FPS fluctuates a lot due to slow resets.
    # also depends a lot on whether levels are in level cache or not
    DmLabSpec('dmlab_benchmark_slow_reset', 'contributed/dmlab30/rooms_keys_doors_puzzle'),

    DmLabSpec('dmlab_sparse', 'contributed/dmlab30/explore_goal_locations_large'),
    DmLabSpec('dmlab_very_sparse', 'contributed/dmlab30/explore_goal_locations_large', extra_cfg={'minGoalDistance': '10'}),
    DmLabSpec('dmlab_sparse_doors', 'contributed/dmlab30/explore_obstructed_goals_large'),
    DmLabSpec('dmlab_nonmatch', 'contributed/dmlab30/rooms_select_nonmatching_object'),
    DmLabSpec('dmlab_watermaze', 'contributed/dmlab30/rooms_watermaze'),

    DmLabSpec('dmlab_skymaze_irreversible_path_hard', 'contributed/dmlab30/skymaze_irreversible_path_hard'),
    DmLabSpec('dmlab_language_select_located_object', 'contributed/dmlab30/language_select_located_object'),
]


def dmlab_env_by_name(name):
    for spec in DMLAB_ENVS:
        if spec.name == name:
            return spec
    raise Exception('Unknown DMLab env')


def get_task_id(env_config):
    if env_config is None:
        return 0
    else:
        num_envs = dmlab30_num_envs()
        return env_config['env_id'] % num_envs


def task_id_to_level(task_id, spec_name):
    if spec_name == 'dmlab_30':
        assert 0 <= task_id < dmlab30_num_envs()
        level_name = tuple(LEVEL_MAPPING.keys())[task_id]
        return level_name_to_level(level_name)
    else:
        raise NotImplementedError(f'Unknown env spec {spec_name}')


# noinspection PyUnusedLocal
def make_dmlab_env_impl(spec, cfg, env_config, **kwargs):
    skip_frames = cfg.env_frameskip

    gpu_idx = 0
    if len(cfg.dmlab_gpus) > 0:
        if kwargs.get('env_config') is not None:
            vector_index = kwargs['env_config']['vector_index']
            gpu_idx = cfg.dmlab_gpus[vector_index % len(cfg.dmlab_gpus)]
            log.debug('Using GPU %d for DMLab rendering!', gpu_idx)

    if spec.level is None:
        task_id = get_task_id(env_config)
        level = task_id_to_level(task_id, spec.name)
    else:
        task_id = 0
        level = spec.level

    env = DmlabGymEnv(
        task_id, level, skip_frames, cfg.res_w, cfg.res_h, cfg.dmlab_throughput_benchmark, cfg.dmlab_renderer,
        get_dataset_path(cfg), cfg.dmlab_with_instructions, cfg.dmlab_extended_action_set,
        cfg.dmlab_use_level_cache, gpu_idx, spec.extra_cfg,
    )

    if env_config:
        env.seed(env_config['env_id'])

    if 'record_to' in cfg and cfg.record_to is not None:
        env = RecordingWrapper(env, cfg.record_to)

    if cfg.pixel_format == 'CHW':
        env = PixelFormatChwWrapper(env)

    env = DmlabRewardShapingWrapper(env)
    return env


def list_all_levels_for_experiment(env_name):
    spec = dmlab_env_by_name(env_name)
    if spec.level is not None:
        level = spec.level
        return [level]
    else:
        if spec.name == 'dmlab_30':
            all_train_levels = list(LEVEL_MAPPING.keys())
            all_train_levels = [level_name_to_level(l) for l in all_train_levels]
            return all_train_levels
        else:
            raise NotImplementedError(f'Unknown env spec {spec.name}')


def make_dmlab_env(env_name, cfg=None, **kwargs):
    ensure_initialized(cfg, env_name)

    spec = dmlab_env_by_name(env_name)
    return make_dmlab_env_impl(spec, cfg=cfg, **kwargs)


def ensure_initialized(cfg, env_name):
    global DMLAB_INITIALIZED
    if DMLAB_INITIALIZED:
        return

    dmlab_register_models()

    all_levels = list_all_levels_for_experiment(env_name)
    dmlab_ensure_global_cache_initialized(experiment_dir(cfg=cfg), all_levels)

    DMLAB_INITIALIZED = True
