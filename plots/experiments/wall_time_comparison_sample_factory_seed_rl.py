import csv
import os
import pickle
import sys
from os.path import join
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plots.plot_utils import BLUE, ORANGE, set_matplotlib_params
from utils.utils import log, ensure_dir_exists

set_matplotlib_params()

# plt.rcParams['figure.figsize'] = (1.5*8.20, 3) #(2.5, 2.0) 7.5， 4
plt.rcParams['figure.figsize'] = (5.5, 3.4) #(2.5, 2.0) 7.5， 4


SAMPLE_FACTORY = 'SampleFactory'
SEED_RL = 'SeedRL'

ALGO_NAME = {SAMPLE_FACTORY: 'SampleFactory APPO', SEED_RL: 'SeedRL V-trace'}


REW_KEY = {
    SAMPLE_FACTORY: '0_aux/avg_true_reward',
    SEED_RL: 'episode_raw_return',
}

COLOR_FRAMEWORK = {
    SAMPLE_FACTORY: ORANGE,
    SEED_RL: BLUE,
}


ENVS_LIST = [
    ('doom_my_way_home', None),
    ('center', None),
]

PLOT_NAMES = dict(
    doom_my_way_home='VizDoom, Find My Way Home',
    center='VizDoom, Defend the Center',
)

FOLDER_NAME = 'aggregates'
hide_file = [f for f in os.listdir(os.getcwd()) if not f.startswith('.') and not f.endswith('.py')]


def interpolate_with_fixed_x_ticks(x, y, x_ticks):
    idx = 0
    interpolated_y = []

    assert len(x) == len(y)
    for x_idx in x_ticks:
        while idx < len(x) - 1 and x[idx] < x_idx:
            idx += 1

        if x_idx == 0:
            interpolated_value = y[idx]
        elif idx < len(y) - 1:
            interpolated_value = (y[idx] + y[idx + 1]) / 2
        else:
            interpolated_value = y[idx]

        interpolated_y.append(interpolated_value)

    return interpolated_y


def extract(experiments, framework):
    # scalar_accumulators = [EventAccumulator(str(dpath / dname / subpath)).Reload().scalars
    #                        for dname in os.listdir(dpath) if dname != FOLDER_NAME and dname in hide_file]

    scalar_accumulators = [EventAccumulator(experiment_dir).Reload().scalars for experiment_dir in experiments]

    log.debug('Event Accumulator finished!')

    # Filter non event files
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    # zhehui sorted(scalar_accumulator.Keys())
    all_keys = [tuple(sorted(scalar_accumulator.Keys())) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    keys = all_keys[0]

    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys]

    # Get and validate all steps per key
    # sorted(all_scalar_events) sorted(scalar_events)
    x_per_key = [[tuple(scalar_event.step for scalar_event in sorted(scalar_events)) for scalar_events in sorted(all_scalar_events)]
                         for all_scalar_events in all_scalar_events_per_key]

    # zhehui
    # import linear interpolation
    # all_steps_per_key = tuple(step_id*1e6 for step_id in range(1e8/1e6))

    # modify_all_steps_per_key = tuple(int(step_id*1e6) for step_id in range(1, int(1e8/1e6 + 1)))
    plot_step = int(5e5)
    all_steps_per_key = [[tuple(int(step_id) for step_id in range(0, int(1e8) + plot_step, plot_step)) for scalar_events in sorted(all_scalar_events)]
                         for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get and average wall times per step per key
    wall_times_per_key = [[tuple(scalar_event.wall_time for scalar_event in scalar_events) for scalar_events in all_scalar_events] for all_scalar_events in all_scalar_events_per_key]

    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    true_reward_key = REW_KEY[framework]
    key_idx = keys.index(true_reward_key)
    values = values_per_key[key_idx]

    x_ticks = steps_per_key[key_idx]
    x_steps = x_per_key[key_idx]

    interpolated_y = []

    for i in range(len(values)):  # outer loop over experiments
        interpolated_y.append(interpolate_with_fixed_x_ticks(x_steps[i], values[i], x_ticks))
        assert len(interpolated_y[i]) == len(x_ticks)

    log.debug('%r', interpolated_y[0][:30])

    times = wall_times_per_key[key_idx]
    for i in range(len(times)):
        times[i] = list(times[i])

    for time_list in times:
        t0 = time_list[0]
        for i in range(len(time_list)):
            time_list[i] -= t0

    max_t_seconds = max(t[-1] for t in times)
    log.debug('max seconds: %.3f', max_t_seconds)
    time_ticks = list(range(0, int(max_t_seconds) + 1, 5))

    time_interpolated_y = []

    for i in range(len(values)):  # outer loop over experiments
        time_interpolated_y.append(interpolate_with_fixed_x_ticks(times[i], values[i], time_ticks))
        assert len(time_interpolated_y[i]) == len(time_ticks)

    return x_ticks, interpolated_y, time_ticks, time_interpolated_y


def aggregate(env, experiments, framework):
    print('Started aggregation {}'.format(env))

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = join(curr_dir, 'cache')
    cache_env = join(cache_dir, f'{env}_{framework}')

    with_cache = False
    if with_cache:
        if os.path.isdir(cache_env):
            with open(join(cache_env, f'{env}.pickle'), 'rb') as fobj:
                interpolated_keys = pickle.load(fobj)
        else:
            cache_env = ensure_dir_exists(cache_env)
            interpolated_keys = extract(experiments, framework)
            with open(join(cache_env, f'{env}.pickle'), 'wb') as fobj:
                pickle.dump(interpolated_keys, fobj)
    else:
        interpolated_keys = extract(experiments, framework)

    return interpolated_keys


def extract_data_tensorboard_events(path, framework):
    all_experiment_dirs = set()
    for filename in path.rglob('*.tfevents.*'):
        experiment_dir = os.path.dirname(filename)
        all_experiment_dirs.add(experiment_dir)

    all_experiment_dirs_list = sorted(list(all_experiment_dirs))
    for experiment_dir in all_experiment_dirs_list:
        log.debug('Experiment dir: %s', experiment_dir)

    log.debug('Total: %d', len(all_experiment_dirs_list))

    experiments_by_env = dict()
    for experiment_dir in all_experiment_dirs_list:
        for env, does_not_contain in ENVS_LIST:
            if env not in experiments_by_env:
                experiments_by_env[env] = []

            if env in experiment_dir and (does_not_contain is None or does_not_contain not in experiment_dir):
                experiments_by_env[env].append(experiment_dir)

    for env, experiments in experiments_by_env.items():
        log.debug('Env %s, experiments: %d (%r)', env, len(experiments), experiments)

    xs_ys_by_env = dict()
    for env, experiments in experiments_by_env.items():
        xs_ys_by_env[env] = aggregate(env, experiments, framework)

    return xs_ys_by_env


def aggregate_csv(env, experiments, framework):
    wall_times = []
    xs = []
    ys = []

    for experiment in experiments:
        with open(experiment) as fp:
            reader = csv.reader(fp, delimiter=',', quotechar='"')
            data_read = [row for row in reader][1:]
            wall_time = []
            x = []
            y = []

            for row in data_read:
                wall_time.append(float(row[0]))
                x.append(float(row[1]))
                y.append(float(row[2]))

            # lowpass avg filter
            y_filt = []
            window = 3
            for i in range(len(y)):
                y_filt.append(np.mean(y[max(0, i - window):min(i + window, len(y))]))

            wall_times.append(wall_time)
            xs.append(x)
            ys.append(y_filt)

    plot_step = int(5e5)
    x_ticks = list(step_id for step_id in range(0, int(1e8) + plot_step, plot_step))

    interpolated_y = []

    for i in range(len(ys)):  # outer loop over experiments
        interpolated_y.append(interpolate_with_fixed_x_ticks(xs[i], ys[i], x_ticks))
        assert len(interpolated_y[i]) == len(x_ticks)

    times = wall_times
    for i in range(len(times)):
        times[i] = list(times[i])

    for time_list in times:
        t0 = time_list[0]
        for i in range(len(time_list)):
            time_list[i] -= t0

    max_t_seconds = max(t[-1] for t in times)
    log.debug('max seconds: %.3f', max_t_seconds)
    time_ticks = list(range(0, int(max_t_seconds) + 1, 15))

    time_interpolated_y = []

    for i in range(len(ys)):  # outer loop over experiments
        time_interpolated_y.append(interpolate_with_fixed_x_ticks(times[i], ys[i], time_ticks))
        assert len(time_interpolated_y[i]) == len(time_ticks)

    return x_ticks, interpolated_y, time_ticks, time_interpolated_y


def extract_data_csv(path, framework):
    all_csvs = list(path.rglob('*.csv'))

    experiments_by_env = dict()
    for csv_file_path in all_csvs:
        csv_file_path = str(csv_file_path)

        for env, does_not_contain in ENVS_LIST:
            if env not in experiments_by_env:
                experiments_by_env[env] = []

            csv_filename = os.path.basename(csv_file_path)
            if env in csv_filename and (does_not_contain is None or does_not_contain not in csv_filename):
                experiments_by_env[env].append(csv_file_path)

    xs_ys_by_env = dict()
    for env, experiments in experiments_by_env.items():
        xs_ys_by_env[env] = aggregate_csv(env, experiments, framework)

    return xs_ys_by_env


def plot(env, interpolated_key, top_ax, bottom_ax, count, framework):
    # zhehui
    # set title
    title_text = PLOT_NAMES[env]
    top_ax.set_title(title_text, fontsize=8)

    x, y, times, y_times = interpolated_key

    y_np = [np.array(yi) for yi in y]
    y_np = np.stack(y_np)

    y_mean = np.mean(y_np, axis=0)
    y_std = np.std(y_np, axis=0)
    y_plus_std = y_mean + y_std
    y_minus_std = y_mean - y_std

    # Configuration
    # fig, ax = plt.subplots()

    def mkfunc(x, pos):
        if x >= 1e6:
            return '%dM' % int(x * 1e-6)
        elif x >= 1e3:
            return '%dK' % int(x * 1e-3)
        else:
            return '%d' % int(x)

    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    top_ax.xaxis.set_major_formatter(mkformatter)

    for ax in [top_ax, bottom_ax]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)

        # hide tick of axis
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        ax.tick_params(which='major', length=0)

        ax.grid(color='#B3B3B3', linestyle='-', linewidth=0.25, alpha=0.2)

        ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))

    # xlabel_text = env.replace('_', ' ').title()
    # plt.xlabel(xlabel_text, fontsize=8)
    # zhehui
    # if they are bottom plots, add Environment Frames
    # if i == 1:
    top_ax.set_xlabel('Env. frames, skip=4', fontsize=8)
    if count == 0:
        top_ax.set_ylabel('Average return', fontsize=8)

    x_delta = 0.05 * x[-1]
    top_ax.set_xlim(xmin=-x_delta, xmax=x[-1] + x_delta)

    y_delta = 0.07 * np.max(y_mean)
    top_ax.set_ylim(ymin=min(np.min(y_mean) - y_delta, 0.0), ymax=np.max(y_plus_std) + y_delta)

    lw = 1.4

    algo_name = ALGO_NAME[framework]
    color = COLOR_FRAMEWORK[framework]

    sf_plot, = top_ax.plot(x, y_mean, color=color, linewidth=lw, antialiased=True)
    top_ax.fill_between(x, y_minus_std, y_plus_std, color=color, alpha=0.25, antialiased=True, linewidth=0.0)

    # top_ax.legend(prop={'size': 6}, loc='lower right')

    #########################################################################################################
    # build the bottom plot (wall time)

    y_np = [np.array(yi) for yi in y_times]
    y_np = np.stack(y_np)

    y_mean = np.mean(y_np, axis=0)
    y_std = np.std(y_np, axis=0)
    y_plus_std = y_mean + y_std
    y_minus_std = y_mean - y_std

    bottom_ax.set_xlabel('Training time, minutes', fontsize=8)
    if count == 0:
        bottom_ax.set_ylabel('Average return', fontsize=8)

    minutes = np.array(times) / 60

    x_delta = 0.05 * minutes[-1]
    bottom_ax.set_xlim(xmin=-x_delta, xmax=minutes[-1] + x_delta)

    y_delta = 0.07 * np.max(y_mean)
    bottom_ax.set_ylim(ymin=min(np.min(y_mean) - y_delta, 0.0), ymax=np.max(y_plus_std) + y_delta)

    bottom_ax.set_xticks([0, 10, 20, 30, 40, 50])

    label = algo_name if count == 1 else None
    bottom_ax.plot(minutes, y_mean, color=color, label=label, linewidth=lw, antialiased=True)
    bottom_ax.fill_between(minutes, y_minus_std, y_plus_std, color=color, alpha=0.25, antialiased=True, linewidth=0.0)

    if count == 1:
        bottom_ax.legend(prop={'size': 6}, loc='lower right')


def plot_envs(interpolated_keys_by_env, top_ax, bottom_ax, framework):
    count = 0
    for env, interpolated_keys in interpolated_keys_by_env.items():
        plot(env, interpolated_keys, top_ax[count], bottom_ax[count], count, framework)
        count += 1


def main():
    sample_factory_runs = '/home/alex/all/projects/sample-factory/train_dir/paper_doom_wall_time_v97_fs4'
    sample_factory_runs_path = Path(sample_factory_runs)

    seed_rl_runs = '/home/alex/all/projects/sample-factory/train_dir/seedrl/seed_rl_csv'
    seed_rl_runs_path = Path(seed_rl_runs)

    fig, (top_ax, bottom_ax) = plt.subplots(2, 2)

    interpolated_keys_by_env = extract_data_tensorboard_events(sample_factory_runs_path, SAMPLE_FACTORY)
    plot_envs(interpolated_keys_by_env, top_ax, bottom_ax, SAMPLE_FACTORY)

    interpolated_keys_by_env = extract_data_csv(seed_rl_runs_path, SEED_RL)
    plot_envs(interpolated_keys_by_env, top_ax, bottom_ax, SEED_RL)

    # plt.show()
    # plot_name = f'{env}_{key.replace("/", " ")}'
    plt.tight_layout()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=1, wspace=0)
    plt.subplots_adjust(wspace=0.12, hspace=0.4)

    plt.margins(0, 0)
    plot_name = f'wall_time'
    plt.savefig(os.path.join(os.getcwd(), f'../final_plots/reward_{plot_name}.pdf'), format='pdf', bbox_inches='tight', pad_inches=0)

    return 0


if __name__ == '__main__':
    sys.exit(main())
