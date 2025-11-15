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
