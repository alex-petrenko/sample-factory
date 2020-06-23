import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plots.plot_utils import set_matplotlib_params
from utils.utils import ensure_dir_exists

# Fixing random state for reproducibility
#np.random.seed(19680801)
DMLAB_30 = {
    145.858254: dict(name='language_select_described_object', sample_factory=145.858254, impala=157.6168929),
    122.7776876: dict(name='language_answer_quantitative_question', sample_factory=122.7776876, impala=146.7571644),
    91.60601354: dict(name='language_select_located_object', sample_factory=91.60601354, impala=105.4298643),
    130.3263547: dict(name='explore_goal_locations_small', sample_factory=130.3263547, impala=102.5641026),
    93.14999496: dict(name='rooms_collect_good_objects_train', sample_factory=93.14999496, impala=92.30769231),
    127.5798033: dict(name='explore_obstructed_goals_small', sample_factory=127.5798033, impala=88.98944193),
    104.2751323: dict(name='explore_object_locations_small', sample_factory=104.2751323, impala=87.4811463),
    88.75805632: dict(name='explore_object_locations_large', sample_factory=88.75805632, impala=79.0346908),
    68.41075306: dict(name='natlab_varying_map_randomized', sample_factory=68.41075306, impala=68.77828054),
    62.53669447: dict(name='natlab_varying_map_regrowth', sample_factory=62.53669447, impala=64.25339367),
    81.16116221: dict(name='explore_goal_locations_large', sample_factory=81.16116221, impala=63.49924585),
    75.14757464: dict(name='explore_obstructed_goals_large', sample_factory=75.14757464, impala=61.23680241),
    57.81540734: dict(name='explore_object_rewards_many', sample_factory=57.81540734, impala=58.37104072),
    35.8566136:  dict(name='rooms_watermaze', sample_factory=35.8566136, impala=55.80693816),
    49.59824968: dict(name='rooms_select_nonmatching_object', sample_factory=49.59824968, impala=55.20361991),
    57.26063443: dict(name='explore_object_rewards_few', sample_factory=57.26063443, impala=51.58371041),
    52.30387768: dict(name='psychlab_continuous_recognition', sample_factory=52.30387768, impala=50.98039216),
    0.3410786613:dict(name='lasertag_three_opponents_small', sample_factory=0.3410786613, impala=37.25490196),
    47.42990654: dict(name='skymaze_irreversible_path_varied', sample_factory=47.42990654, impala=34.6907994),
    45.39414074: dict(name='rooms_keys_doors_puzzle', sample_factory=45.39414074, impala=34.83709273),
    22.87701212: dict(name='natlab_fixed_large_map', sample_factory=22.87701212, impala=33.20802005),
    31.93193193: dict(name='skymaze_irreversible_path_hard', sample_factory=31.93193193, impala=31.07769424),
    51.79391333: dict(name='psychlab_arbitrary_visuomotor_mapping', sample_factory=51.79391333, impala=30.32581454),
    43.90400394: dict(name='rooms_exploit_deferred_effects_train', sample_factory=43.90400394, impala=26.06516291),
    0.0696158471:dict(name='lasertag_three_opponents_large', sample_factory=0.0696158471, impala=20.80200501),
    22.17931013: dict(name='language_execute_random_task', sample_factory=22.17931013, impala=20.1754386),
    0.9826005903:dict(name='lasertag_one_opponent_small', sample_factory=0.9826005903, impala=19.04761905),
    0.0:         dict(name='lasertag_one_opponent_large', sample_factory=0.0, impala=6.140350877),
    100.3156284: dict(name='psychlab_visual_search', sample_factory=100.3156284, impala=0),
    76.46110081: dict(name='psychlab_sequential_comparison', sample_factory=76.46110081, impala=0),
}

set_matplotlib_params()
plt.rcParams['figure.figsize'] = (10, 12)

# plt.rcdefaults()
plt.rc('axes', axisbelow=True)
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.tick_params(which='major', length=0)
ax.grid(axis='x', linestyle='-.')

ax.set_ylim(ymin=-0.3, ymax=29.7)

# Example data
y_pos = np.arange(len(DMLAB_30))

yticklabels = []
sample_factory_performance = []
impala_performance = []

for i in sorted(DMLAB_30.keys(), reverse=True):
    yticklabels.append(DMLAB_30[i]['name'])
    sample_factory_performance.append(DMLAB_30[i]['sample_factory'])
    impala_performance.append(DMLAB_30[i]['impala'])


sample_factory_performance = np.array(sample_factory_performance)
impala_performance = np.array(impala_performance)
width = 0.4
ax.barh(y_pos, sample_factory_performance, width, align='center', color='#ff7f0e', label='Sample Factory')
ax.barh(y_pos + width, impala_performance, width, align='center', color='#1f77b4', label='DeepMind IMPALA')
# ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), ncol=2, loc='lower left', mode='expend')
handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels, bbox_to_anchor=(0., 1.01, 0.7, 0.3), loc='lower left', ncol=2, mode='expand', frameon=False, fontsize=12)

#
# lgd.set_in_layout(True)

ax.set_yticks(y_pos)
ax.set_yticklabels(yticklabels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.axvline(100, color='#b3b3b3')
ax.set_xlabel('Human Normalised Score, %')




# plt.show()
plt.tight_layout()
plot_name = 'dmlab_30_score'
plot_dir = ensure_dir_exists(os.path.join(os.getcwd(), 'final_plots'))
plt.savefig(os.path.join(plot_dir, f'{plot_name}.pdf'), format='pdf')