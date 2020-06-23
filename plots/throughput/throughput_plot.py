import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# sns.set()
from plots.plot_utils import BLUE, DARK_GREY, set_matplotlib_params
from utils.utils import ensure_dir_exists

system_1_xticks = [0, 100, 200, 300, 400, 500, 600, 700]
system_1_xmax = 700

system_2_xticks = [0, 500, 1000, 1500, 2000]
system_2_xmax = 2000

measurements = {
    'system_1_atari': dict(filename='10_core_atari.csv', x_ticks=system_1_xticks,y_ticks=[10000, 20000, 30000, 40000, 50000], x_max=system_1_xmax, y_max=51000,y_label=False),
    'system_1_doom': dict(filename='10_core_vizdoom.csv', x_ticks=system_1_xticks, y_ticks=[10000, 20000, 30000, 40000, 50000, 60000], x_max=system_1_xmax, y_max=61000, y_label=False),
    'system_1_dmlab': dict(filename='10_core_dmlab.csv', x_ticks=system_1_xticks, y_ticks=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000], x_max=system_1_xmax, y_max=16000, y_label=False),
    'system_2_atari': dict(filename='36_core_atari.csv', x_ticks=system_2_xticks,y_ticks=[20000, 40000, 60000, 80000, 100000, 120000, 140000], x_max=system_2_xmax,y_max=143000, y_label=False),
    'system_2_doom': dict(filename='36_core_vizdoom.csv', x_ticks=system_2_xticks, y_ticks=[20000, 40000, 60000, 80000, 100000, 120000, 140000], x_max=system_2_xmax, y_max=150000, y_label=False),
    'system_2_dmlab': dict(filename='36_core_dmlab.csv', x_ticks=system_2_xticks, y_ticks=[10000, 20000, 30000, 40000, 50000], x_max=system_2_xmax, y_max=50000, y_label=False),
}

titles = {
    'system_1_atari': 'Atari throughput, System #1',
    'system_1_doom': 'VizDoom throughput, System #1',
    'system_1_dmlab': 'DMLab throughput, System #1',
    'system_2_atari': 'Atari throughput, System #2',
    'system_2_doom': 'VizDoom throughput, System #2',
    'system_2_dmlab': 'DMLab throughput, System #2',
}

set_matplotlib_params()

plt.rcParams['figure.figsize'] = (10.0, 4.0) #(2.5, 2.0) 7.5， 4

def build_plot(name, measurement, ax, count):
    # data = pd.read_csv(join('data', measurement['filename']))
    data = pd.read_csv(measurement['filename'])

    x = data.values[:, 0]

    # Sample Factory

    sf_y = data.values[:, 1]

    # rlpyt
    rlpyt_y = data.values[:, 2]

    # rllib
    rllib_y_p1 = data.values[:, 3]

    # rllib_x_p2 = rllib.values[4::, 0]
    # rllib_y_p2 = rllib.values[4::, 1]

    # scalable_agent
    sa_y_p1 = data.values[:, 4].astype('float')
    # sa_x_p2 = scalable_agent.values[2::, 0]
    # sa_y_p2 = scalable_agent.values[2::, 1]

    # seed-rl
    seedrl_y_p1 = data.values[:, 5]

    mkfunc = lambda x, pos: '%1.1fM' % (x * 1e-6) if x >= 1e6 else '%dK' % int(x * 1e-3) if x >= 1e3 else '%d' % int(x)
    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    ax.yaxis.set_major_formatter(mkformatter)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)

    # ax.set_axisbelow(True)
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # Title and label
    title = titles[name]
    ax.set_title(title, fontsize=8)
    if count >= 3:
        ax.set_xlabel('Num. environments', fontsize=8)

    if count %3 == 0:
        ax.set_ylabel('FPS, frameskip = 4', fontsize=8)

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
    ax.grid(color='#B3B3B3', linestyle='--', linewidth=0.25, alpha=0.2, dashes=(15, 10))
    ax.xaxis.grid(False)
    ax.set_xlim(xmin=0, xmax=measurement['x_max'])
    ax.set_ylim(ymin=0, ymax=measurement['y_max'])
    # plt.grid(False)

    ax.ticklabel_format(style='plain', axis='x', scilimits=(0, 0))
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(5, 5))

    ax.set_xticks(measurement['x_ticks'])
    ax.set_yticks(measurement['y_ticks'])

    # plot each line

    # color {'b', 'g', 'r', 'c', 'm'}
    # Not good {'y', 'k', 'w'}
    # sample factory
    marker_size = 3.3
    lw = 2.0

    sf_plot, = ax.plot(x, sf_y, color='#FF7F0E', label='SampleFactory APPO', marker="o", markersize=marker_size, linewidth=lw)

    # seed-rl
    seed_rl, = ax.plot(x, seedrl_y_p1, color=BLUE, label='SeedRL V-trace', marker="o", markersize=marker_size, linewidth=lw)  # label='scalable_agent',

    # plt.plot(rllib_x_p1, rllib_y_p1,  color='skyblue', label='rllib',marker="o")
    rllib_p1, = ax.plot(x, rllib_y_p1, color='#2CA02C', label='RLlib IMPALA', marker="o", markersize=marker_size, linewidth=lw)
    # rllib_p2, = plt.plot(rllib_x_p2, rllib_y_p2, color='#2CA02C', marker='x', markersize=marker_size_cross, linestyle=":")

    # scalable_agent
    sa_p1, = ax.plot(x, sa_y_p1, color='#d62728', label='DeepMind IMPALA', marker="o", markersize=marker_size, linewidth=lw)  # label='scalable_agent',
    # sa_p2, = plt.plot(sa_x_p2, sa_y_p2, color='#d62728', marker="x", markersize=marker_size_cross, linestyle=":")

    # rlpyt
    rlpyt_plot, = ax.plot(x, rlpyt_y, color=DARK_GREY, label='rlpyt PPO', marker="o", markersize=marker_size, linewidth=lw)


    # plot legend
    # sa_legend = plt.legend([sf_plot, rlpyt_plot, (rllib_p1, rllib_p2), (sa_p1, sa_p2)],
    #                        ['SampleFactory', 'rlpyt', 'rllib', 'IMPALA'], numpoints=1,
    #                        handler_map={tuple: HandlerTuple(ndivide=None)}, prop={'size': 7})
    # sa_legend.get_frame().set_linewidth(0.25)


def main():
    # requirements
    # 1) dark background
    # 2) both axis should start at 0
    # 3) Legend should be on background
    # 4) Legend should not obstruct data
    # 5) Export in eps
    # 6) Markers. Little circles for every data point
    # 7) Dashed lines for missing data
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    count = 0

    ax = (ax1, ax2, ax3, ax4, ax5, ax6)
    for name, measurement in measurements.items():
        build_plot(name, measurement, ax[count], count)
        count += 1

    handles, labels = ax[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center')
    # lgd = fig.legend(handles, labels, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand")
    lgd = fig.legend(handles, labels, bbox_to_anchor=(0.05, 0.88, 0.9, 0.5), loc='lower left', ncol=5, mode="expand")
    lgd.set_in_layout(True)

    # plt.show()
    plot_name = 'throughput'
    # plt.subplots_adjust(wspace=0.05, hspace=0.15)
    # plt.margins(0, 0)
    # plt.tight_layout(rect=(0, 0, 1, 1.2))
    # plt.subplots_adjust(bottom=0.2)

    plt.tight_layout(rect=(0, 0, 1.0, 0.9))
    # plt.show()

    plot_dir = ensure_dir_exists(os.path.join(os.getcwd(), '../final_plots'))

    plt.savefig(os.path.join(plot_dir, f'../final_plots/{plot_name}.pdf'), format='pdf', bbox_extra_artists=(lgd,))
    # plt.savefig(os.path.join(os.getcwd(), f'../final_plots/{plot_name}.pdf'), format='pdf', bbox_inches='tight', pad_inches=0, bbox_extra_artists=(lgd,))


if __name__ == '__main__':
    sys.exit(main())
