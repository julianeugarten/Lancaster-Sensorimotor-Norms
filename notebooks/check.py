# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import time

# make time stamp
ts = time.strftime("%Y-%m-%d_%H-%M")
print(f"Timestamp: {ts}")

# %%
df = pd.read_json("../data/fanfics_metadata_with_sensorimotor_scores.json", orient='records', lines=True)

drop_cols = ['total_foot_leg.mean','normalized_foot_leg.mean', 'avg_matched_foot_leg.mean','total_hand_arm.mean', 'normalized_hand_arm.mean',
       'avg_matched_hand_arm.mean', 'total_head.mean', 'normalized_head.mean','avg_matched_head.mean', 'total_mouth.mean', 'normalized_mouth.mean',
       'avg_matched_mouth.mean', 'total_torso.mean', 'normalized_torso.mean','avg_matched_torso.mean']
df = df.drop(columns=drop_cols)

# add the sensitivity columns
sensitivity_df = pd.read_json("../data/2025-11-18_12-39_fanfics_sensitivity_labelled.json", orient='records', lines=True)
sensitivity_labels = [x for x in sensitivity_df.columns if x.startswith('sensitive_')] + ["work_id", "sensitivity_prop_above_threshold"] # just get the important columns
sensitivity_df = sensitivity_df[sensitivity_labels]
df = df.merge(sensitivity_df, how='left', on='work_id')
df.head()

# %%

##### sense columns and derived metrics #####

# decide which columns to use, here senses + normalized & engagement metrics
sense_cols = ["auditory.mean", "gustatory.mean", "olfactory.mean", "haptic.mean", "visual.mean", "interoceptive.mean"]
use_what = "normalized_" # set this to total, avg_matched, or normalized
sense_cols_prefixed = [use_what + col for col in sense_cols]

# make a sense-ratio column where all senses sum to 1
df[f'{use_what}sense_sum'] = df[sense_cols_prefixed].sum(axis=1)
# and an overall sum
total_cols = [col for col in df.columns if col.startswith("total_")]
df['sense_overall_sum'] = df[total_cols].mean(axis=1)
# and we add, for each sense, how big a percent of the total sense intensity it is
for sense in sense_cols_prefixed:
    colname = sense.replace('.mean', '_percent')
    df[f"{colname}"] = df[sense] / df[f'{use_what}sense_sum']

# entropy of the sense distribution (we use the percent columns for this, since they sum to 1)
df['sense_entropy'] = df[[col for col in df.columns if col.endswith('_percent')]].apply(lambda x: entropy(x, base=2), axis=1)
df.head()

# define additional sense columns to use
add_sense_cols = [f'{use_what}sense_sum', 'sense_overall_sum', 'sense_entropy'] 
percent_sense_cols = [col for col in df.columns if col.endswith('_percent')] # percent show some of the same info so we skip them for now
# and define the sensitivity labels
sensitive_cols = [col for col in df.columns if col.startswith('sensitive_')] + ['sensitivity_prop_above_threshold']
# %%


# %%

# formalize the columns we want to look at
engagement_cols = ['kudos_hits_ratio', 'comment_hits_ratio', 'kudos_ratio_resid', 'comment_ratio_resid', 'maturity_rating', 'days_since_published']

corr_cols = sense_cols_prefixed + percent_sense_cols + add_sense_cols + sensitive_cols + engagement_cols
# heatmap of correlations
plt.figure(figsize=(17, 15))
corr = df[corr_cols].corr(method='spearman')
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, square=True)
plt.title('Correlation Heatmap between Sensorimotor Scores and Engagement Metrics')
plt.savefig(f"../figs/{ts}_correlation_heatmap_based_on_{use_what}.png", bbox_inches='tight')
plt.show()

# %%
corr

# %%

# plot the relationship between months since published and kudos_hits_ratio with regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x='days_since_published', y='kudos_hits_ratio', data=df, alpha=0.5)
sns.lineplot(x='days_since_published', y=model.fittedvalues, color='red', data=df)
plt.title('Kudos-Hits Ratio vs. Days Since Published with Regression Line')
plt.xlabel('Days Since Published')
plt.ylabel('Kudos-Hits Ratio')
plt.show()

# lets see the comment_hits_ratio vs months since published
plt.figure(figsize=(8, 6))
sns.scatterplot(x='days_since_published', y='comment_hits_ratio', data=df, alpha=0.5)
sns.lineplot(x='days_since_published', y=model_comments.fittedvalues, color='red', data=df)
plt.title('Comment-Hits Ratio vs. Days Since Published with Regression Line')
plt.xlabel('Days Since Published')
plt.ylabel('Comment-Hits Ratio')
plt.show()

# make historgrams of the residuals
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['kudos_ratio_resid'].dropna(), kde=True)
plt.title('Histogram of Kudos-Hits Ratio Residuals')
plt.subplot(1, 2, 2)
sns.histplot(df['comment_ratio_resid'].dropna(), kde=True)
plt.title('Histogram of Comment-Hits Ratio Residuals')
plt.show()
# %%