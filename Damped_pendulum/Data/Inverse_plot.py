# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:49
# @Author  : Jay
# @File    : Inverse_plot.py
# @Project: Damped_pendulum
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_newton = pd.read_csv('../Data/Newton_Inverse.csv')
df_onsager = pd.read_csv('../Data/Onsager_Inverse.csv')
df_eit = pd.read_csv('../Data/EIT_Inverse.csv')

noise_levels = sorted(df_newton["noise"].unique())
n_noise = len(noise_levels)

# Prepare grouped data
data_models = {
    "Newton": [np.abs(df_newton[df_newton["noise"] == n]["lamb_value"].values) for n in noise_levels],
    "Onsager": [np.abs(df_onsager[df_onsager["noise"] == n]["lamb_value"].values) for n in noise_levels],
    "EIT": [np.abs(df_eit[df_eit["noise"] == n]["lamb_value"].values) for n in noise_levels],
}

colors = {
    "Newton": "violet",
    "Onsager": "slateblue",
    "EIT": "salmon"
}

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['svg.fonttype'] = 'none'

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, (model, data) in enumerate(data_models.items()):
    ax = axes[i]
    vp = ax.violinplot(data, positions=np.arange(n_noise),
                       showmeans=True, showextrema=True, showmedians=True)

    for partname in ('bodies', 'cmeans', 'cmaxes', 'cmins', 'cbars', 'cmedians'):
        if partname == 'bodies':
            for body in vp['bodies']:
                body.set_facecolor(colors[model])
                body.set_edgecolor('black')
                body.set_alpha(0.7)
        else:
            vp[partname].set_color(colors[model])
            vp[partname].set_linewidth(2.0)

    ax.axhline(0.20, linestyle='--', color='black', linewidth=1)

    ax.set_title(model, fontsize=20, fontweight='bold')
    ax.set_xticks(np.arange(n_noise))
    ax.set_xticklabels(noise_levels, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    if i == 0:
        ax.set_ylabel("Learned lamb", fontsize=20, fontweight='bold')
        ax.set_xlabel("Noise", fontsize=20, fontweight='bold')

plt.tight_layout()
plt.show()
fig.savefig('figure_4.svg', dpi=600)
