


# %%%
# Distributions of senses across len
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from colorblind_sim import simulate_plot_with_daltonlens

from daltonize import daltonize

# %%
CWD = pathlib.Path.cwd().parent
RESOURCES_DIR = CWD / "resources"
FIGS = CWD / "figs"
FIGS.mkdir(exist_ok=True)

# %%
# for each word, how many senses have = 0.0
# get our data
norms = pd.read_csv(RESOURCES_DIR / "Lancaster_sensorimotor_norms_for_39707_words.csv")
norms.head()

# %%

senses = [x for x in norms.columns if x.endswith('.mean')]
senses = senses[:6]
# for word in norms, we get n > 0.0 per sense and sum
senses_dict = {sense: {"sum": 0, "count": 0, "mean": 0} for sense in senses}
for sense in senses:
    sum_all = norms[sense].sum()
    count_all = len(norms.loc[norms[sense] > 0.0])
    mean_all = norms[sense].mean() 
    means_nonzero = norms.loc[norms[sense] > 0.0][sense].mean()
    senses_dict[sense]["sum"] = sum_all
    senses_dict[sense]["count"] = count_all
    senses_dict[sense]["mean"] = mean_all
    senses_dict[sense]["mean_nonzero"] = means_nonzero

senses_dict


# %%


index = np.arange(len(senses))
bar_width = 0.4

# get colorblind friendly palette
palette = sns.color_palette("colorblind", n_colors=10)

sns.set_style("whitegrid")

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(5.5, 6.5),
    sharey=True,
    gridspec_kw={'wspace': 0.0})

sum_values = [senses_dict[sense]["sum"] * -1 for sense in senses]
count_values = [senses_dict[sense]["count"] * -1 for sense in senses]
mean_values = [senses_dict[sense]["mean"] for sense in senses]

# left side
ax1.barh(index, sum_values, bar_width, label='Sum', color=palette[4])
ax1.barh([i + bar_width for i in index], count_values, bar_width, label='N > 0.0', color=palette[8])
ax1.set_xlabel("Score")
ax1.legend(loc="lower left")
# rotate x-axis labels to avoid overlap
plt.setp(ax1.get_xticklabels(), rotation=30, ha="center")
# make x ticks positive
ax1.set_xticklabels([abs(int(x)) for x in ax1.get_xticks()])


# right side
ax2.barh(index, mean_values, bar_width, color="black", label="Mean", align="edge")
#ax2.barh([i + bar_width for i in index], [senses_dict[sense]["mean_nonzero"] for sense in senses], bar_width, color="lightgray", label="Mean (non-zero)")

ax2.set_xlabel("Mean Value")
ax2.set_yticks(index)
ax2.set_yticklabels([s.split(".")[0] for s in senses])
ax2.legend(loc="lower right")
plt.setp(ax2.get_xticklabels(), rotation=30, ha="center")

plt.tight_layout()
plt.show()


# %%

# get internal correlations between senses in the dictionary
correlations = norms[senses].corr(method="spearman")

palette = sns.color_palette("cividis", n_colors=6)

fig = plt.figure(figsize=(4, 4))
sns.heatmap(correlations, annot=True, cmap=palette, vmin=-1, vmax=1, cbar=False)
simulate_plot_with_daltonlens(lambda: fig)

# %%

dist = 1 - correlations

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

Z = linkage(squareform(dist), method="average")
order = leaves_list(Z)

# see the hierarchical clustering dendrogram
from scipy.cluster.hierarchy import dendrogram
plt.figure(figsize=(5, 3.5))
dendrogram(Z, labels=[s.split(".")[0] for s in senses], leaf_rotation=0, link_color_func=lambda x: "black")
# rotate x-axis labels to avoid overlap
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha="center")
plt.ylabel("Distance (1 - Spearman Correlation)")
plt.tight_layout()
plt.show()

ordered_senses = list(np.array(senses)[order])

palette = sns.color_palette("cividis", n_colors=6)
fig = plt.figure(figsize=(4, 4))
sns.heatmap(correlations.loc[ordered_senses, ordered_senses], annot=True, cmap=palette, vmin=-1, vmax=1, cbar=False)
# set labels to be the sense names without the .mean suffix
plt.yticks(np.arange(len(ordered_senses)) + 0.5, [s.split(".")[0] for s in ordered_senses])
plt.xticks(np.arange(len(ordered_senses)) + 0.5, [s.split(".")[0] for s in ordered_senses])
plt.tight_layout()
# fig.savefig(FIGS / "sense_correlations.png", dpi=300, bbox_inches="tight")
plt.show()

# daltonize the figure to see how it looks under different color vision deficiencies
sim_fig = daltonize.simulate_mpl(fig, copy=True)
daltonized_fig = daltonize.daltonize_mpl(fig, copy=True)

plt.show()

ordered_senses
# %%
sim_fig.show()
# %%
