import ctypes
import multiprocessing
import os
import random
import shutil
from os.path import join
from pathlib import Path

from sample_factory.utils.utils import ensure_dir_exists, log, safe_ensure_dir_exists


LEVEL_SEEDS_FILE_EXT = 'dm_lvl_seeds'

# we normally don't need more than 30 tasks (for DMLab-30), or 57 tasks for Atari
MAX_NUM_TASKS = 64


def filename_to_level(filename):
    level = filename.split('.')[0]
    level = level[1:]  # remove leading underscore
    return level


def level_to_filename(level):
    # add leading underscore so these folders appear at the top of cache dir (to simplify debugging)
    filename = f'_{level}.{LEVEL_SEEDS_FILE_EXT}'
    return filename


def read_seeds_file(filename, has_keys):
    seeds = []

    with open(filename, 'r') as seed_file:
        lines = seed_file.readlines()
        for line in lines:
            try:
                if has_keys:
                    seed, cache_key = line.split(' ')
                else:
                    seed = line

                seed = int(seed)
                seeds.append(seed)
            except Exception:
                log.error('Could not read seed value from the file! File potentially corrupted')
                log.exception('Exception when reading seeds file')

    return seeds


class DmlabLevelCacheGlobal:
    """
    This is a global DMLab level cache designed to be shared across multiple worker processes.
    Here's how it works:

    1) We pre-generate a number of maps for environments that require level generation by either running the
    experiment without caching, or by using a specifically designed script. The generated levels are
    stored in the cache directory, and the correspondence from seed to the level is stored in separate *.dm_lvl_seeds
    files, e.g. a list of seeds present in the cache for each of the environments

    2) When we start a new experiment, we read all the available *.dm_lvl_seeds files in the cache dir. As long as
    we have available pre-generated levels we will only use seeds that correspond to these levels to avoid costly
    generation of new levels. The index of the last used seed is stored in shared memory and protected with a lock,
    to guarantee that different processes won't use the same seed.

    3) When we run out of available seeds we're just going to use new random seeds. Levels generated from these random
    seeds will be added to the cache folder, as well as to the *.dm_lvl_seeds files.

    4) Every time the level from cache is used we record it in another *.dm_lvl_seeds file in the experiment folder.
    This allows us to read this file when we restart the experiment, to guarantee that we won't be reusing seeds.

    The order of pre-calculated seeds is currently randomized on each run.
    This class works in conjunction with fetch/write methods of the DMLab Gym env (see dmlab_gym.py)

    """

    def __init__(self,  cache_dir, experiment_dir, all_levels_for_experiment, policy_idx):
        self.cache_dir = cache_dir
        self.experiment_dir = experiment_dir
        self.policy_idx = policy_idx

        self.all_seeds = dict()
        self.available_seeds = dict()
        self.used_seeds = dict()
        self.num_seeds_used_in_current_run = dict()
        self.locks = dict()

        for lvl in all_levels_for_experiment:
            self.all_seeds[lvl] = []
            self.available_seeds[lvl] = []
            self.num_seeds_used_in_current_run[lvl] = multiprocessing.RawValue(ctypes.c_int32, 0)
            self.locks[lvl] = multiprocessing.Lock()

        log.debug('Reading the DMLab level cache...')
        cache_dir = ensure_dir_exists(cache_dir)

        lvl_seed_files = Path(os.path.join(cache_dir, '_contributed')).rglob(f'*.{LEVEL_SEEDS_FILE_EXT}')
        for lvl_seed_file in lvl_seed_files:
            lvl_seed_file = str(lvl_seed_file)
            level = filename_to_level(os.path.relpath(lvl_seed_file, cache_dir))
            self.all_seeds[level] = read_seeds_file(lvl_seed_file, has_keys=True)
            self.all_seeds[level] = list(set(self.all_seeds[level]))  # leave only unique seeds
            log.debug('Level %s has %d total seeds available', level, len(self.all_seeds[level]))

        log.debug('Updating level cache for the current experiment...')
        used_lvl_seeds_dir = self.get_used_seeds_dir()
        used_seeds_files = Path(used_lvl_seeds_dir).rglob(f'*.{LEVEL_SEEDS_FILE_EXT}')
        self.used_seeds = dict()
        for used_seeds_file in used_seeds_files:
            used_seeds_file = str(used_seeds_file)
            level = filename_to_level(os.path.relpath(used_seeds_file, used_lvl_seeds_dir))
            self.used_seeds[level] = read_seeds_file(used_seeds_file, has_keys=False)
            log.debug('%d seeds already used in this experiment for level %s', len(self.used_seeds[level]), level)

            self.used_seeds[level] = set(self.used_seeds[level])

        for lvl in all_levels_for_experiment:
            lvl_seeds = self.all_seeds.get(lvl, [])
            lvl_used_seeds = self.used_seeds.get(lvl, [])

            lvl_remaining_seeds = set(lvl_seeds) - set(lvl_used_seeds)
            self.available_seeds[lvl] = list(lvl_remaining_seeds)

            same_levels_for_population = False
            if same_levels_for_population:
                # shuffle with fixed seed so agents in population get the same levels
                random.Random(42).shuffle(self.available_seeds[lvl])
            else:
                random.shuffle(self.available_seeds[lvl])

            log.debug('Env %s has %d remaining unused seeds', lvl, len(self.available_seeds[lvl]))

        log.debug('Done initializing global DMLab level cache!')

    def get_used_seeds_dir(self):
        return ensure_dir_exists(join(self.experiment_dir, f'dmlab_used_lvl_seeds_p{self.policy_idx:02d}'))

    def record_used_seed(self, level, seed):
        self.num_seeds_used_in_current_run[level].value += 1
        log.debug('Updated number of used seeds for level %s (%d)', level, self.num_seeds_used_in_current_run[level].value)

        used_lvl_seeds_dir = self.get_used_seeds_dir()
        used_seeds_filename = join(used_lvl_seeds_dir, level_to_filename(level))
        safe_ensure_dir_exists(os.path.dirname(used_seeds_filename))

        with open(used_seeds_filename, 'a') as fobj:
            fobj.write(f'{seed}\n')

        # this data structure is not shared across processes, but we mostly care about the initial
        # seeds anyway, which are initialized before the processes are forked
        if level not in self.used_seeds:
            self.used_seeds[level] = {seed}
        else:
            self.used_seeds[level].add(seed)

    def get_unused_seed(self, level, random_state=None):
        with self.locks[level]:
            num_used_seeds = self.num_seeds_used_in_current_run[level].value
            if num_used_seeds >= len(self.available_seeds.get(level, [])):
                # we exhaused all the available pre-calculated levels, let's generate a new random seed!

                while True:
                    if random_state is not None:
                        new_seed = random_state.randint(0, 2 ** 31 - 1)
                    else:
                        new_seed = random.randint(0, 2 ** 31 - 1)

                    if level not in self.used_seeds:
                        break

                    if new_seed in self.used_seeds[level]:
                        # log.debug('Random seed %d already used in the past!', new_seed)
                        pass
                    else:
                        break
            else:
                new_seed = self.available_seeds[level][num_used_seeds]

            self.record_used_seed(level, new_seed)
            return new_seed

    def add_new_level(self, level, seed, key, pk3_path):
        with self.locks[level]:
            num_used_seeds = self.num_seeds_used_in_current_run[level].value
            if num_used_seeds < len(self.available_seeds.get(level, [])):
                log.warning('We should only add new levels to cache if we ran out of pre-generated levels (seeds)')
                log.warning(
                    'Num used seeds: %d, available seeds: %d, level: %s, seed %r, key %r',
                    num_used_seeds, len(self.available_seeds.get(level, [])), level, seed, key,
                )

                # some DMLab-30 environments, e.g. language_select_located_object may require different levels even
                # for the same seed. This is most likely a bug in DeepMind Lab, because the same seed should generate
                # identical environments

            path = os.path.join(self.cache_dir, key)
            if not os.path.isfile(path):
                # copy the cached file DeepMind Lab has written to the cache directory
                shutil.copyfile(pk3_path, path)

            # add new map to the list of available seeds for this level
            # so it can be used next time we run the experiment
            lvl_seeds_filename = join(self.cache_dir, level_to_filename(level))
            safe_ensure_dir_exists(os.path.dirname(lvl_seeds_filename))
            with open(lvl_seeds_filename, 'a') as fobj:
                fobj.write(f'{seed} {key}\n')

            # we're not updating self.all_seeds and self.available_seeds here because they are not shared across processes
            # basically the fact that we're adding a new level means that we ran out of cache and we won't need it
            # anymore in this experiment


def dmlab_ensure_global_cache_initialized(experiment_dir, all_levels_for_experiment, num_policies, level_cache_dir):
    global DMLAB_GLOBAL_LEVEL_CACHE

    assert multiprocessing.current_process().name == 'MainProcess', \
        'make sure you initialize DMLab cache before child processes are forked'

    DMLAB_GLOBAL_LEVEL_CACHE = []
    for policy_id in range(num_policies):
        # level cache is of course shared between independently training policies
        # it's easiest to achieve

        log.info('Initializing level cache for policy %d...', policy_id)
        cache = DmlabLevelCacheGlobal(level_cache_dir, experiment_dir, all_levels_for_experiment, policy_id)
        DMLAB_GLOBAL_LEVEL_CACHE.append(cache)
