# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_json("../data/fanfics_metadata_with_sensorimotor_scores.json", orient='records', lines=True)
df.head()
# %%
df.columns
# %%
col_normalized = [col for col in df.columns if col.startswith('total_')]

df['kudos_hits_ratio'] = df['kudos'] / df['hits'].replace(0, 1)  # avoid division by zero
corr_cols = col_normalized + ['comments', 'kudos', 'hits', 'kudos_hits_ratio']
# heatmap of correlations
plt.figure(figsize=(10, 8))
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, square=True)
plt.title('Correlation Heatmap between Sensorimotor Scores and Engagement Metrics')
plt.show()

# %%
df.columns
# %%
# okay, i think what we want are ratios over a book for all senses
sense_cols = [
    "auditory.mean", "gustatory.mean", "olfactory.mean",
    "haptic.mean", "visual.mean", "interoceptive.mean"
]

use_what = "normalized_"

sense_cols_prefixed = [use_what + col for col in sense_cols]

# make a sense-ratio column where all senses sum to 1
df['sense_total'] = df[sense_cols_prefixed].sum(axis=1)

for sense in sense_cols_prefixed:
    colname = sense.replace('.mean', '_percent')
    df[f"{colname}"] = df[sense] / df['sense_total']
df.head()

# %%
# now heatmap of correlations again
corr_cols = [col for col in df.columns if col.endswith('_percent')] + ['comments', 'kudos', 'hits', 'kudos_hits_ratio']
plt.figure(figsize=(10, 8))
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, square=True)
plt.title('Correlation Heatmap between Sense Ratios and Engagement Metrics')
plt.show()
# %%
