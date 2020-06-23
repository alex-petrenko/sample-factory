import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

labels = ['20', '40', '80', '160', '320', '640']
# fps_by_method_10_core_cpu = dict(
#     deepmind_impala=[8590, 10596, 10941, 10928, 13328, math.nan],
#     rllib_appo=[9384, 9676, 11171, 11328, 11590, 11345],
#     ours=[11565, 16982, 25068, 37410, 46977, 52033]
# )
# data = fps_by_method_10_core_cpu

fps_by_method_36_core_cpu = dict(
    deepmind_impala=[6951, 8191, 8041, 9900, 10014, math.nan],
    rllib_appo=[13308, 23608, 30568, 31002, 32840, 33784],
    ours=[11586, 20410, 33326, 46243, 70124, 86753],
)
data = fps_by_method_36_core_cpu

# ours: 160=40x4, 320=40x8 with 3072 bs, 640=80x8 with 3072 bs

# multi-policy:
# 2 policies, 640 actors, 93284 FPS
# 4 policies: 1600 actors, 116320 FPS
# 8 policies: 1600 actors,

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))

item_idx = 0
bars = dict()
for key, value in data.items():
    rects = ax.bar(x + item_idx * width - len(data) * width / 2, value, width, label=key)
    bars[key] = rects
    item_idx += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Num. environments in parallel')
ax.set_ylabel('Environment frames per second')
ax.set_title('Throughput of different RL methods')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')


autolabel(bars['ours'])

fig.tight_layout()
plt.show()
