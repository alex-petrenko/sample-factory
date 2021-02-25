import argparse
import os
import pickle
import sys
from os.path import join
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plots.plot_utils import BLUE, set_matplotlib_params
from utils.utils import ensure_dir_exists

set_matplotlib_params()

plt.rcParams['figure.figsize'] = (1.5 * 8.20, 5)  # (2.5, 2.0) 7.5， 4

PLOT_NAMES_LIST = ['avg_reward', 'avg_true_reward', 'avg_rew_crash', 'avg_num_collisions', 'avg_rew_orient']

PLOT_KEY = ['0_aux/avg_reward', '0_aux/avg_true_reward', '0_aux/avg_rew_crash', '0_aux/avg_num_collisions',
            '0_aux/avg_rew_orient']

PLOT_STEP = int(20e6)
TOTAL_STEP = int(1e9)

NUM_AGENTS = 8
TIME_METRIC_COLLISION = 60  # ONE MINUTE

def extract(experiments):
    scalar_accumulators = [EventAccumulator(experiment_dir).Reload().scalars for experiment_dir in experiments]

    # Filter non event files
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if
                           scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    all_keys = [tuple(sorted(scalar_accumulator.Keys())) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, \
        "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)

    keys = all_keys[0]
    all_scalar_events_per_key = [[scalar_accumulator.Items(key)
                                  for scalar_accumulator in scalar_accumulators] for key in keys]

    # Get and validate all steps per key
    x_per_key = [[tuple(scalar_event.step
                 for scalar_event in sorted(scalar_events)) for scalar_events in sorted(all_scalar_events)]
                 for all_scalar_events in all_scalar_events_per_key]

    plot_step = PLOT_STEP
    all_steps_per_key = [[tuple(int(step_id) for step_id in range(0, TOTAL_STEP, plot_step))
                          for _ in sorted(all_scalar_events)]
                          for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(
            all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]
    plot_key = PLOT_KEY

    interpolated_keys = dict()
    for tmp_id in range(len(plot_key)):
        key_idx = keys.index(plot_key[tmp_id])
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

                if plot_key[tmp_id] == '0_aux/avg_num_collisions':
                    interpolated_value = interpolated_value * TIME_METRIC_COLLISION / NUM_AGENTS
                    interpolated_value = np.log(interpolated_value)

                interpolated_y[i].append(interpolated_value)
            assert len(interpolated_y[i]) == len(x)

        print(interpolated_y[0][:30])

        interpolated_keys[plot_key[tmp_id]] = (x, interpolated_y)

    return interpolated_keys


def aggregate(path, subpath, experiments, ax):
    print("Started aggregation {}".format(path))

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = join(curr_dir, 'cache')
    cache_env = join(cache_dir, subpath)

    if os.path.isdir(cache_env):
        with open(join(cache_env, f'{subpath}.pickle'), 'rb') as fobj:
            interpolated_keys = pickle.load(fobj)
    else:
        cache_env = ensure_dir_exists(cache_env)
        interpolated_keys = extract(experiments=experiments)
        with open(join(cache_env, f'{subpath}.pickle'), 'wb') as fobj:
            pickle.dump(interpolated_keys, fobj)

    for i, key in enumerate(interpolated_keys.keys()):
        plot(i, interpolated_keys[key], ax[i])


# def plot(env, key, interpolated_key, ax, count):
def plot(index, interpolated_key, ax):
    # set title
    title_text = PLOT_NAMES_LIST[index]
    ax.set_title(title_text, fontsize=8)

    x, y = interpolated_key
    y_np = [np.array(yi) for yi in y]
    y_np = np.stack(y_np)
    y_mean = np.mean(y_np, axis=0)
    y_std = np.std(y_np, axis=0)
    y_plus_std = y_mean + y_std
    y_minus_std = y_mean - y_std

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

    # if they are bottom plots, add Environment Frames
    if index == 0 or index >=3:
        ax.set_xlabel('Env. frames', fontsize=8)
    if index == 0:
        ax.set_ylabel('Average return', fontsize=8)

    # hide tick of axis
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(which='major', length=0)

    ax.grid(color='#B3B3B3', linestyle='-', linewidth=0.25, alpha=0.2)
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))

    marker_size = 0
    lw = 1.4
    lw_baseline = 0.7

    sf_plot, = ax.plot(x, y_mean, color=BLUE, linewidth=lw, antialiased=True)
    ax.fill_between(x, y_minus_std, y_plus_std, color=BLUE, alpha=0.25, antialiased=True, linewidth=0.0)

    # ax.legend(prop={'size': 6}, loc='lower right')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='main path for tensorboard files', default=os.getcwd())
    parser.add_argument('--output', type=str,
                        help='aggregation can be saves as tensorboard file (summary) or as table (csv)', default='csv')

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        raise argparse.ArgumentTypeError('Parameter {} is not a valid path'.format(path))

    subpath = os.listdir(path)[0]
    all_experiment_dirs = []
    for filename in Path(args.path).rglob('*.tfevents.*'):
        experiment_dir = os.path.dirname(filename)
        all_experiment_dirs.append(experiment_dir)

    if args.output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))
    ax1 = plt.subplot(232)
    ax2 = plt.subplot(233)
    ax3 = plt.subplot(235)
    ax4 = plt.subplot(236)
    ax0 = plt.subplot(131)

    ax = (ax0, ax1, ax2, ax3, ax4)
    aggregate(path, subpath, all_experiment_dirs, ax=ax)


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.16, hspace=0.18)
    plt.margins(0, 0)


    # plt.show()
    plot_name = f'test'
    plt.savefig(os.path.join(os.getcwd(), f'../final_plots/reward_{plot_name}.pdf'),
                format='pdf', bbox_inches='tight', pad_inches=0)

    return 0


if __name__ == '__main__':
    sys.exit(main())
