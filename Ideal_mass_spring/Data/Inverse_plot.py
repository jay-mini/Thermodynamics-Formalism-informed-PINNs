import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_newton = pd.read_csv('../Data/Newton_Inverse.csv')
df_onsager = pd.read_csv('../Data/Lagrange_Inverse.csv')
df_eit = pd.read_csv('../Data/Hamilton_Inverse.csv')

noise_levels = sorted(df_newton["noise"].unique())
n_noise = len(noise_levels)

data_models = {
    "Newton": [df_newton[df_newton["noise"] == n]["km_value"].values for n in noise_levels],
    "Lagrange": [df_onsager[df_onsager["noise"] == n]["km_value"].values for n in noise_levels],
    "Hamilton": [df_eit[df_eit["noise"] == n]["km_value"].values for n in noise_levels],
}

colors = {
    "Newton": "violet",
    "Lagrange": "slateblue",
    "Hamilton": "salmon"
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

    ax.axhline(1.0, linestyle='--', color='black', linewidth=1)

    ax.set_title(model, fontsize=20, fontweight='bold')
    ax.set_xticks(np.arange(n_noise))
    ax.set_xticklabels(noise_levels, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    if i == 0:
        ax.set_ylabel("k/m", fontsize=20, fontweight='bold')
        ax.set_xlabel("Noise", fontsize=20, fontweight='bold')

    for j, values in enumerate(data):
        x_jitter = np.random.normal(loc=j, scale=0.04, size=len(values))
        ax.scatter(
            x_jitter,
            values,
            s=35,
            color=colors[model],
            edgecolors='black',
            linewidth=0.6,
            alpha=0.6,
            zorder=3
        )

plt.tight_layout()
plt.show()
fig.savefig('figure_4.svg', dpi=600)
