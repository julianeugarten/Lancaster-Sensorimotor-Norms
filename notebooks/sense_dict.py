
# %%
# Distributions of senses across len
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# %%
CWD = pathlib.Path.cwd().parent
RESOURCES_DIR = CWD / "resources"

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
