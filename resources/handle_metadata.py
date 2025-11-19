# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import statsmodels.api as sm

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

# fix some cols
df['published'] = pd.to_datetime(df['published'], errors='coerce')
# add time since published column
reference_date = pd.to_datetime("2023-01-01")
diff = reference_date - df['published']
df['days_since_published'] = diff.dt.days

# reception metrics
# check nan in comments and hits
reception_cols = ['comments', 'hits', 'kudos']
for col in reception_cols:
    print(f"Number of NaNs in {col}: {df[col].isna().sum()}")
    # fillna with 0
    df[col] = df[col].fillna(0)

# %%

# we want to look at distributions
sense_cols = [f'normalized_{sense}.mean' for sense in ['auditory', 'gustatory', 'olfactory', 'haptic', 'visual', 'interoceptive']]
engagement_cols = ['kudos', 'comments', 'hits', 'days_since_published']

tmp = df.sample(n=4000, random_state=42) # sample for speed

for col in sense_cols + engagement_cols:
    print(f"Shapiro-Wilk test for {col}:")
    data = tmp[col].dropna()
    stat, p = shapiro(data)
    print(f"  Statistic={stat:.4f}, p-value={p:.4f}")
    plt.figure(figsize=(8, 4))
    sns.set_style("whitegrid")
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# %%

##### engagement metrics #####

# we use ratios since "hits", for example, are very sensitive to time on the platform.
# ratios are not subject to drift in the same way. It’s a conversion rate: how many of the people who saw this actually cared.
df['kudos_hits_ratio'] = df['kudos'] / df['hits'].replace(0, 1) # avoid division by zero
df['comment_hits_ratio'] = df['comments'] / df['hits'].replace(0, 1)
# still, we do see that they are related to time on platform, so we can try to regress out time. Essentially, we want a kudo-ratio without the age-effect.
# age-effect might haver to do with visibility on the platform, or with changing user behavior over time; random stuff like it's not on the top page anymore, etc.
# lets ask: Across the entire dataset, how does kudos/hits typically drift as a function of months since publication?
# we draw that curve (simple linear model) and then take the residuals as our "age-corrected" measure of engagement.
# essentially, we get a number that tells us: given how old this fic is, is it doing better or worse than expected? did it perform above or below the age-norm?
# most fics will be close to zero, some will be strongly positive (doing better than expected) or negative (doing worse than expected).
# simple linear regression to get residuals
# for kudos
# x = sm.add_constant(df['days_since_published']) # adding constant so we know we have a baseline
# y = df['kudos_hits_ratio']
# model = sm.OLS(y, x).fit()
# df['kudos_ratio_resid'] = model.resid
# # same for comments
# x = sm.add_constant(df['days_since_published'])
# y = df['comment_hits_ratio']
# model_comments = sm.OLS(y, x).fit()
# df['comment_ratio_resid'] = model_comments.resid

# linear regression on non-NaN rows
valid_kudos = df['kudos_hits_ratio'].notna()
x = sm.add_constant(df.loc[valid_kudos, 'days_since_published'])
y = df.loc[valid_kudos, 'kudos_hits_ratio']
model = sm.OLS(y, x).fit()
df.loc[valid_kudos, 'kudos_ratio_resid'] = model.resid

valid_comments = df['comment_hits_ratio'].notna()
x = sm.add_constant(df.loc[valid_comments, 'days_since_published'])
y = df.loc[valid_comments, 'comment_hits_ratio']
model_comments = sm.OLS(y, x).fit()
df.loc[valid_comments, 'comment_ratio_resid'] = model_comments.resid

# add the maturity rating as a numeric code
rating_map = {
    'General Audiences': 0,
    'Teen And Up Audiences': 1,
    'Mature': 2,
    'Explicit': 3,
    'Not Rated': np.nan} # so higher number means more mature content
df['maturity_rating'] = df['rating'].map(rating_map)

