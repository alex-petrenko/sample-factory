import pickle
import sys

from os.path import join

import matplotlib
import matplotlib.pyplot as plt

from plots.plot_utils import set_matplotlib_params
from utils.utils import log, ensure_dir_exists

import ast
import argparse
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event

set_matplotlib_params()


ENVS_LIST = [
    ('doom_my_way_home', None),
    ('doom_deadly_corridor', None),
    ('defend_the_center', None),
    ('doom_defend_the_line', None),
    ('health_gathering', 'supreme'),
    ('health_gathering_supreme', None),
]

FOLDER_NAME = 'aggregates'
hide_file = [f for f in os.listdir(os.getcwd()) if not f.startswith('.') and not f.endswith('.py')]


def extract(experiments):
    # scalar_accumulators = [EventAccumulator(str(dpath / dname / subpath)).Reload().scalars
    #                        for dname in os.listdir(dpath) if dname != FOLDER_NAME and dname in hide_file]

    scalar_accumulators = [EventAccumulator(experiment_dir).Reload().scalars for experiment_dir in experiments]

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
    plot_step = int(2.5e6)
    all_steps_per_key = [[tuple(int(step_id) for step_id in range(0, int(5e8), plot_step)) for scalar_events in sorted(all_scalar_events)]
                         for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get and average wall times per step per key
    # wall_times_per_key = [np.mean([tuple(scalar_event.wall_time for scalar_event in scalar_events) for scalar_events in all_scalar_events], axis=0)
    #                       for all_scalar_events in all_scalar_events_per_key]

    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    true_reward_key = '0_aux/avg_true_reward'
    key_idx = keys.index(true_reward_key)
    values = values_per_key[key_idx]

    x = steps_per_key[key_idx]
    x_steps = x_per_key[key_idx]

    interpolated_y = [[] for _ in values]

    for i in range(len(values)):
        idx = 0

        values[i] = values[i][2:]
        x_steps[i] = x_steps[i][2:]

        assert len(x_steps[i]) == len(values[i])
        for x_idx in x:
            while x_steps[i][idx] < x_idx and idx < len(x_steps[i]):
                idx += 1

            if x_idx == 0:
                interpolated_value = values[i][idx]
            elif idx < len(values[i]) - 1:
                interpolated_value = (values[i][idx] + values[i][idx + 1]) / 2
            else:
                interpolated_value = values[i][idx]

            interpolated_y[i].append(interpolated_value)

        assert len(interpolated_y[i]) == len(x)

    print(interpolated_y[0][:30])

    interpolated_keys = dict()
    interpolated_keys[true_reward_key] = (x, interpolated_y)

    return interpolated_keys


def aggregate_to_summary(dpath, aggregation_ops, extracts_per_subpath):
    for op in aggregation_ops:
        for subpath, all_per_key in extracts_per_subpath.items():
            path = dpath / FOLDER_NAME / op.__name__ / dpath.name / subpath
            aggregations_per_key = {key: (steps, wall_times, op(values, axis=0)) for key, (steps, wall_times, values) in all_per_key.items()}
            write_summary(path, aggregations_per_key)


def write_summary(dpath, aggregations_per_key):
    writer = tf.summary.FileWriter(dpath)

    for key, (steps, wall_times, aggregations) in aggregations_per_key.items():
        for step, wall_time, aggregation in zip(steps, wall_times, aggregations):
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=aggregation)])
            scalar_event = Event(wall_time=wall_time, step=step, summary=summary)
            writer.add_event(scalar_event)

        writer.flush()


def aggregate_to_csv(dpath, aggregation_ops, extracts_per_subpath):
    for subpath, all_per_key in extracts_per_subpath.items():
        for key, (steps, values) in all_per_key.items():
            # aggregations = [op(values, axis=0) for op in aggregation_ops]
            aggregations = [value for value in values]
            write_csv(dpath, subpath, key, dpath.name, aggregations, steps, [1,2,3])

    # for subpath, all_per_key in extracts_per_subpath.items():
    #     for key, (steps, wall_times, values) in all_per_key.items():
    #         aggregations = [op(values, axis=0) for op in aggregation_ops]
    #         write_csv(dpath, subpath, key, dpath.name, aggregations, steps, aggregation_ops)


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def write_csv(dpath, subpath, key, fname, aggregations, steps, aggregation_ops):
    path = dpath / FOLDER_NAME

    if not path.exists():
        os.makedirs(path)

    file_name = get_valid_filename(key) + '-' + get_valid_filename(subpath) + '-' + fname + '.csv'
    # aggregation_ops_names = [aggregation_op.__name__ for aggregation_op in aggregation_ops]
    # df = pd.DataFrame(np.transpose(aggregations), index=steps, columns=aggregation_ops_names)
    df = pd.DataFrame(np.transpose(aggregations), index=steps)
    df.to_csv(path / file_name, sep=';')


def aggregate(env, experiments, output):
    aggregation_ops = [np.mean, np.min, np.max, np.median, np.std, np.var]

    ops = {
        'summary': aggregate_to_summary,
        'csv': aggregate_to_csv
    }

    print("Started aggregation {}".format(env))

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = join(curr_dir, 'cache')
    cache_env = join(cache_dir, env)

    if os.path.isdir(cache_env):
        with open(join(cache_env, f'{env}.pickle'), 'rb') as fobj:
            interpolated_keys = pickle.load(fobj)
    else:
        cache_env = ensure_dir_exists(cache_env)
        interpolated_keys = extract(experiments)
        with open(join(cache_env, f'{env}.pickle'), 'wb') as fobj:
            pickle.dump(interpolated_keys, fobj)

    for key in interpolated_keys.keys():
        plot(env, key, interpolated_keys[key])


def plot(env, key, interpolated_key):
    plt.rcParams['figure.figsize'] = (2.5, 2.0)

    x, y = interpolated_key

    y_np = [np.array(yi) for yi in y]
    y_np = np.stack(y_np)

    y_mean = np.mean(y_np, axis=0)
    y_std = np.std(y_np, axis=0)
    y_plus_std = np.minimum(y_mean + y_std, y_np.max())
    y_minus_std = y_mean - y_std

    # Configuration
    fig, ax = plt.subplots()

    #
    def mkfunc(x, pos):
        if x >= 1e6:
            return '%dM' % int(x * 1e-6)
        elif x >= 1e3:
            return '%dK' % int(x * 1e-3)
        else:
            return '%d' % int(x)

    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    ax.xaxis.set_major_formatter(mkformatter)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)

    # ax.set_axisbelow(True)
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # Title and label
    # title = titles[name]
    # ax.set_title(title, fontsize=8)

    xlabel_text = env.replace('_', ' ').title()
    plt.xlabel(xlabel_text, fontsize=8)

    # if measurement['y_label']:
    #     plt.ylabel('FPS, frameskip = 4', fontsize=8)

    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # hide tick of axis
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(which='major', length=0)

    # use logarithmic for x axis! NOT A GOOD IDEA
    # plt.xscale('symlog')

    # let trim and label seems light
    # ax.tick_params(colors='gray', direction='out')
    # for tick in ax.get_xticklabels():
    #     tick.set_color('gray')
    # for tick in ax.get_yticklabels():
    #     tick.set_color('gray')

    # let plot a little bit larger
    # draw dash gray grid lines
    # plt.grid(color='#B3B3B3', linestyle='--', linewidth=0.25, alpha=0.2, dashes=(15, 10))
    plt.grid(color='#B3B3B3', linestyle='-', linewidth=0.25, alpha=0.2)
    # ax.xaxis.grid(False)

    x_delta = 0.05 * x[-1]
    plt.xlim(xmin=-x_delta, xmax=x[-1] + x_delta)

    y_delta = 0.06 * np.max(y_mean)
    plt.ylim(ymin=min(np.min(y_mean) - y_delta, 0.0), ymax=np.max(y_mean) + y_delta)
    # plt.grid(False)

    # plt.ticklabel_format(style='sci', axis='x', scilimits=(8, 8))
    plt.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))

    # plt.xticks(measurement['x_ticks'])
    # plt.yticks(measurement['y_ticks'])

    # plot each line

    # color {'b', 'g', 'r', 'c', 'm'}
    # Not good {'y', 'k', 'w'}
    # sample factory
    marker_size = 0
    lw = 1.4

    sf_plot, = plt.plot(x, y_mean, color='#FF7F0E', label='SampleFactory', linewidth=lw, antialiased=True)
    plt.fill_between(x, y_minus_std, y_plus_std, color='#FF7F0E', alpha=0.25, antialiased=True, linewidth=0.0)

    # plot legend
    # sa_legend = plt.legend([sf_plot, rlpyt_plot, (rllib_p1, rllib_p2), (sa_p1, sa_p2)],
    #                        ['SampleFactory', 'rlpyt', 'rllib', 'IMPALA'], numpoints=1,
    #                        handler_map={tuple: HandlerTuple(ndivide=None)}, prop={'size': 7})
    # sa_legend.get_frame().set_linewidth(0.25)

    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=1, wspace=0)
    plt.margins(0, 0)

    # plt.show()
    plot_name = f'{env}_{key.replace("/", " ")}'
    plt.savefig(os.path.join(os.getcwd(), f'../final_plots/reward_{plot_name}.pdf'), format='pdf', bbox_inches='tight', pad_inches=0)


def main():
    def param_list(param):
        p_list = ast.literal_eval(param)
        if type(p_list) is not list:
            raise argparse.ArgumentTypeError("Parameter {} is not a list".format(param))
        return p_list

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="main path for tensorboard files", default=os.getcwd())
    parser.add_argument("--subpaths", type=param_list, help="subpath sturctures", default=['test', 'train'])
    parser.add_argument("--output", type=str, help="aggregation can be saves as tensorboard file (summary) or as table (csv)", default='csv')

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(path))

    # hide_file = [f for f in os.listdir(path) if not f.startswith('.') and not f.endswith('.py')]
    # subpaths = [path / dname / subpath for subpath in args.subpaths for dname in os.listdir(path) if dname != FOLDER_NAME and dname in hide_file]

    all_experiment_dirs = set()
    for filename in Path(args.path).rglob('*.tfevents.*'):
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

    # for subpath in subpaths:
    #     if not os.path.exists(subpath):
    #         raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(subpath))

    if args.output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    for env, experiments in experiments_by_env.items():
        aggregate(env, experiments, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
