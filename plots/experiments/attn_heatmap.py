import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from plots.plot_utils import set_matplotlib_params

set_matplotlib_params()

PAGE_WIDTH_INCHES = 8.2
FULL_PAGE_WIDTH = 1.4 * PAGE_WIDTH_INCHES
HALF_PAGE_WIDTH = FULL_PAGE_WIDTH / 2

plt.rcParams['figure.figsize'] = (HALF_PAGE_WIDTH, 2.1)  # (2.5, 2.0) 7.5， 4

def main():
    attention_scores = np.array([
        [0, 0.18558, 0.19735, 0.61707],
        [0.37036, 0, 0.29203, 0.33761],
        [0.37889, 0.30201, 0, 0.31910],
        [0.57469, 0.224756, 0.17775, 0],
    ])
    attention_scores_no_vel = np.array([
        [0, 0.39367, 0.29643, 0.30989],
        [0.36004, 0, 0.32138, 0.31858],
        [0.32372, 0.30816, 0, 0.36811],
        [0.34186, 0.33828, 0.31986, 0]
    ])
    cmap = sns.cm.rocket_r
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.heatmap(attention_scores, ax=ax1, linewidth=0.5, cmap="Reds", vmin=0, vmax=0.66)
    ax1.set_title("Attention weights")
    sns.heatmap(attention_scores_no_vel, ax=ax2, linewidths=0.5, cmap="Reds",vmin=0, vmax=0.66)
    ax2.set_title("Attention weights, velocity = 0")
    fig.tight_layout()
    axes = (ax1, ax2)
    plt.setp(axes, xticks=np.arange(4) + 0.5, xticklabels=['red', 'grey', 'green', 'blue'], yticks=np.arange(4) + 0.5,
             yticklabels=['red', 'grey', 'green', 'blue'])
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), f'attn_study.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.02)
if __name__ == '__main__':
    main()