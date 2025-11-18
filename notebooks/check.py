# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import entropy


# %%
df = pd.read_json("../data/fanfics_metadata_with_sensorimotor_scores.json", orient='records', lines=True)

drop_cols = ['total_foot_leg.mean','normalized_foot_leg.mean', 'avg_matched_foot_leg.mean','total_hand_arm.mean', 'normalized_hand_arm.mean',
       'avg_matched_hand_arm.mean', 'total_head.mean', 'normalized_head.mean','avg_matched_head.mean', 'total_mouth.mean', 'normalized_mouth.mean',
       'avg_matched_mouth.mean', 'total_torso.mean', 'normalized_torso.mean','avg_matched_torso.mean']
df = df.drop(columns=drop_cols)

# add the sensitivity columns
sensitivity_df = pd.read_json("../data/2025-11-17_14-11_fanfics_sensitivity_labelled.json", orient='records', lines=True)
sensitivity_labels = [x for x in sensitivity_df.columns if x.startswith('sensitive_')] + ["work_id"] # just get the important columns
sensitivity_df = sensitivity_df[sensitivity_labels]
df = df.merge(sensitivity_df, how='left', on='work_id')
df.head()

# %%
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

# and add entropy
def calculate_entropy(values):
    values = np.array(values)
    total = values.sum()
    if total == 0:
        # uniform distribution if all zeros
        values = np.ones_like(values) / len(values)
    else:
        values = values / total
    return entropy(values, base=2)  # optional: base=2 for bits

df['sense_entropy'] = df[sense_cols_prefixed].apply(calculate_entropy, axis=1)
df.head()

add_sense_cols = [f'{use_what}sense_sum', 'sense_overall_sum', 'sense_entropy'] #+ [col for col in df.columns if col.endswith('_percent')] # percent show some of the same info so we skip them for now
# %%
# fix some cols
df['published'] = pd.to_datetime(df['published'], errors='coerce')
# add time since published column
reference_date = pd.to_datetime("2023-01-01")
diff = reference_date - df['published']
df['days_since_published'] = diff.dt.days

# reception metrics
# we use ratios since "hits", for example, are very sensitive to time on the platform.
# ratios are not subject to drift in the same way. It’s a conversion rate: how many of the people who saw this actually cared.
df['kudos_hits_ratio'] = df['kudos'] / df['hits'].replace(0, np.nan) # avoid division by zero
df['comment_hits_ratio'] = df['comments'] / df['hits'].replace(0, np.nan)
# still, we do see that they are related to time on platform, so we can try to regress out time. Essentially, we want a kudo-ratio without the age-effect.
# age-effect might haver to do with visibility on the platform, or with changing user behavior over time; random stuff like it's not on the top page anymore, etc.
# lets ask: Across the entire dataset, how does kudos/hits typically drift as a function of months since publication?
# we draw that curve (simple linear model) and then take the residuals as our "age-corrected" measure of engagement.
# essentially, we get a number that tells us: given how old this fic is, is it doing better or worse than expected? did it perform above or below the age-norm?
# most fics will be close to zero, some will be strongly positive (doing better than expected) or negative (doing worse than expected).
# simple linear regression to get residuals
# for kudos
x = sm.add_constant(df['days_since_published']) # adding constant so we know we have a baseline
y = df['kudos_hits_ratio']
model = sm.OLS(y, x, missing='drop').fit()
df['kudos_ratio_resid'] = model.resid
# same for comments
x = sm.add_constant(df['days_since_published'])
y = df['comment_hits_ratio']
model_comments = sm.OLS(y, x, missing='drop').fit()
df['comment_ratio_resid'] = model_comments.resid

# add the maturity rating as a numeric code
rating_map = {
    'General Audiences': 0,
    'Teen And Up Audiences': 1,
    'Mature': 2,
    'Explicit': 3,
    'Not Rated': np.nan} # so higher number means more mature content
df['rating_code'] = df['rating'].map(rating_map)

# formalize the columns we want to look at
engagement_cols = ['kudos_hits_ratio', 'comment_hits_ratio', 'kudos_ratio_resid', 'comment_ratio_resid', 'rating_code', 'days_since_published']

corr_cols = sense_cols_prefixed + add_sense_cols + engagement_cols
# heatmap of correlations
plt.figure(figsize=(15, 10))
corr = df[corr_cols].corr(method='spearman')
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, square=True)
plt.title('Correlation Heatmap between Sensorimotor Scores and Engagement Metrics')
plt.savefig("../figs/correlation_heatmap.png", bbox_inches='tight')
plt.show()

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