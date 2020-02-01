import os
import sys
from os.path import join
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.legend_handler import HandlerTuple

from plots.plot_utils import copy_plot_themes

AXES_COLOR = '#E6E6E6'

# sns.set()

def main():
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.legend_handler import HandlerTuple
    import os
    from matplotlib import rc
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 12
    plt.rcParams["figure.figsize"] = (10, 6)
    # requirements
    # 1) dark background
    # 2) both axis should start at 0
    # 3) Legend should be on background
    # 4) Legend should not obstruct data
    # 5) Export in eps
    # 6) Markers. Little circles for every data point
    # 7) Dashed lines for missing data

    # Load the data
    sample_factory = pd.read_csv('sample-factory.csv')
    rlpyt = pd.read_csv('rlpyt.csv')
    rllib = pd.read_csv('rllib.csv')
    scalable_agent = pd.read_csv('scalable_agent.csv')

    # Sample Factory
    sf_x = sample_factory.values[:, 0]
    sf_y = sample_factory.values[:, 1]

    # rlpyt
    rlpyt_x = rlpyt.values[:, 0]
    rlpyt_y = rlpyt.values[:, 1]

    # rllib
    rllib_x_p1 = rllib.values[0:5:, 0]
    rllib_y_p1 = rllib.values[0:5, 1]

    rllib_x_p2 = rllib.values[4::, 0]
    rllib_y_p2 = rllib.values[4::, 1]

    # scalable_agent
    sa_x_p1 = scalable_agent.values[0:3:, 0]
    sa_y_p1 = scalable_agent.values[0:3:, 1]

    sa_x_p2 = scalable_agent.values[2::, 0]
    sa_y_p2 = scalable_agent.values[2::, 1]

    # Configuration
    ax = plt.axes()
    # ax.set_axisbelow(True)
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # Title and label
    plt.title('VizDoom Throughput, FPS', fontsize=20)
    plt.xlabel('Total num envs', fontsize=16)
    plt.ylabel('fps, frameskip$=4$', fontsize=16)

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
    plt.grid(color='#B3B3B3', linestyle='--', linewidth=1, alpha=0.3)
    plt.xlim(xmin=0, xmax=1800)
    plt.ylim(ymin=0, ymax=140000)

    # plot each line

    # color {'b', 'g', 'r', 'c', 'm'}
    # Not good {'y', 'k', 'w'}
    # sample factory
    sf_plot, = plt.plot(sf_x, sf_y, color='#FF7F0E', label='Sample Factory', marker="o", markersize=3.6)

    # rlpyt
    rlpyt_plot, = plt.plot(rlpyt_x, rlpyt_y, color='#1F77B4', label='rlpyt', marker="o", markersize=3.6)

    # plt.plot(rllib_x_p1, rllib_y_p1,  color='skyblue', label='rllib',marker="o")
    rllib_p1, = plt.plot(rllib_x_p1, rllib_y_p1, color='#2CA02C', marker="o", markersize=3.6)
    rllib_p2, = plt.plot(rllib_x_p2, rllib_y_p2, color='#2CA02C', marker='x', markersize=3.4, linestyle=":")

    # scalable_agent
    sa_p1, = plt.plot(sa_x_p1, sa_y_p1, color='#AEC7E8', marker="o", markersize=3.6)  # label='scalable_agent',
    sa_p2, = plt.plot(sa_x_p2, sa_y_p2, color='#AEC7E8', marker="x", markersize=3.4, linestyle=":")

    # plot legend
    sa_legend = plt.legend([sf_plot, rlpyt_plot, (rllib_p1, rllib_p2), (sa_p1, sa_p2)],
                           ['Sample Factory, APPO', 'rlpyt, PPO', 'rllib, IMPALA', 'DeepMind, IMPALA'], numpoints=1,
                           handler_map={tuple: HandlerTuple(ndivide=None)})

    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), "result.eps"), format='eps')


if __name__ == '__main__':
    sys.exit(main())
