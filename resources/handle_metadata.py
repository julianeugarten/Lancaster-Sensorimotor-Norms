# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import statsmodels.api as sm
import statsmodels.formula.api as smf
import time

# timestamp
ts = time.strftime("%Y-%m-%d")
print("Timestamp:", ts)
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

# OBS dropping rows

print("Original length of dataframe:", len(df))
# drop rows with missing sensorimotor scores
sense_cols = [f'normalized_{sense}.mean' for sense in ['auditory', 'gustatory', 'olfactory', 'haptic', 'visual', 'interoceptive']]
engagement_cols = ['kudos', 'comments', 'hits', 'days_since_published']

# drop rows with value == 0 in any sense cols
for col in sense_cols:
    df = df[df[col] != 0]
print("Length after dropping rows with no sensorimotor scores:", len(df))

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

##### engagement metrics #####

# we use ratios since "hits", for example, are very sensitive to time on the platform.
# ratios are not subject to drift in the same way. It’s a conversion rate: how many of the people who saw this actually cared.
df['kudos_hits_ratio'] = df['kudos'] / df['hits']#.replace(0, 1) # avoid division by zero
df['comment_hits_ratio'] = df['comments'] / df['hits']#.replace(0, 1)
# still, we do see that they are related to time on platform, so we can try to regress out time. Essentially, we want a kudo-ratio without the age-effect.
# age-effect might haver to do with visibility on the platform, or with changing user behavior over time; random stuff like it's not on the top page anymore, etc.
# lets ask: Across the entire dataset, how does kudos/hits typically drift as a function of months since publication?
# we draw that curve (simple linear model) and then take the residuals as our "age-corrected" measure of engagement.
# essentially, we get a number that tells us: given how old this fic is, is it doing better or worse than expected? did it perform above or below the age-norm?
# most fics will be close to zero, some will be strongly positive (doing better than expected) or negative (doing worse than expected).
# simple linear regression to get residuals
# for kudos

# print na in engagement cols
print("Number of NaNs in engagement ratios before regression:", df['kudos_hits_ratio'].isna().sum(), df['comment_hits_ratio'].isna().sum())

# we want to look at distributions
tmp = df.sample(n=4000, random_state=42) # sample for speed

# subplot setup
fig, axes = plt.subplots(3, 2, figsize=(15, 9))
axes = axes.flatten()
for i, sense in enumerate(sense_cols):
    # shapiro
    stat, p = shapiro(tmp[sense].dropna())
    print(f'Shapiro-Wilk test for {sense}: stat={stat:.4f}, p={p:.4f}')
    ax = axes[i]
    sns.histplot(df[sense], kde=True, ax=ax)
plt.tight_layout()
plt.savefig(f"../figs/distributions/{ts}_sensorimotor_distributions.png")
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(15, 9))
axes = axes.flatten()
for i, col in enumerate(engagement_cols + ['kudos_hits_ratio', 'comment_hits_ratio']):
    # shapiro
    stat, p = shapiro(tmp[col].dropna())
    print(f'Shapiro-Wilk test for {col}: stat={stat:.4f}, p={p:.4f}')
    ax = axes[i]
    sns.histplot(df[col], kde=True, ax=ax)
plt.tight_layout()
plt.savefig(f"../figs/distributions/{ts}_engagement_distributions.png")
plt.show()

# alright, very long tails for the engagement metrics
# also ratios are quite skewed
# we should log-transform them before regression

# %%

##### Test with simple linear regression first #####

# log transform days since published, as the distribution is highly skewed
df['log_age'] = np.log(df['days_since_published'])
X_log = sm.add_constant(df['log_age'])

engagement_cols = ['kudos_hits_ratio', 'comment_hits_ratio', 'hits', 'kudos', 'comments']

for col in engagement_cols:
    y = df[col]
    # log transform y because of skewness
    y_log = np.log(y.replace(0, 1)) # avoid log(0)
    print(f"\nProcessing OLS for {col}...")
    print(f"OLS summary for {col}:")
    model = sm.OLS(y_log, X_log, missing='drop').fit()
    # model summary
    print(model.summary())
    
    # # store residuals
    # df[f'{col}_resid'] = model.resid

    # # add spearman correlation of residuals with days_since_published
    # print("\n")
    # print(f"spearman of residuals {col} with days_since_published:")
    # corr = df[[f'{col}_resid', 'days_since_published']].dropna().corr(method='spearman').iloc[0,1]
    # print(f"{corr:.4f}")

    # # and lets plot the regression
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(x='days_since_published', y=y_log, data=df, alpha=0.5)
    # sns.lineplot(x='days_since_published', y=model.fittedvalues, color='red', data=df)
    # plt.title(f'Log-Transformed {col} vs. Days Since Published with Regression Line')
    # plt.xlabel('Days Since Published')
    # plt.ylabel(f'Log-Transformed {col}')
    # plt.show()

# %%

#### Mixed-effects model #####
# we want to account for author-level variability, so we use a mixed-effects model with random intercepts for authors

# first, make sure author is categorical
df['author'] = df['author'].astype('category')

# Mixed-effects model loop
for col in engagement_cols:
    print(f"\nProcessing MixedLM for {col} (log-transformed)...")
    model_df = df.copy()

    # Log-transform col
    #df[f'log_{col}'] = np.log(df[col] + 0.01) # add small constant to avoid log(0)
    # we dont just want to add 1 to avoid log(0), because that would distort small values too much
    epsilon = 0.001 * (model_df[col].max() - model_df[col].min())
    model_df[f'log_{col}'] = np.log(model_df[col] + epsilon)
    
    # Drop rows with missing values in dependent, predictor, or group
    model_df = model_df[['author', f'log_{col}', 'log_age', 'days_since_published', 'work_id']].dropna()
    
    # Define and fit the mixed-effects model
    formula = f'log_{col} ~ log_age'
    md = smf.mixedlm(formula, model_df, groups=model_df["author"])
    mdf = md.fit(reml=False)
    print(mdf.summary())
    
    # Store residuals
    model_df[f'{col}_resid'] = mdf.resid
    # save workid and residuals to main df
    df = df.merge(model_df[['work_id', f'{col}_resid']], how='left', on='work_id')
    
    # Check residual correlation with days_since_published
    corr = model_df[[f'{col}_resid', 'days_since_published']].corr(method='spearman').iloc[0,1]
    print(f"Spearman correlation of residuals {col} x days: {corr:.4f}")

    # Plot regression results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='days_since_published', y=model_df[f'log_{col}'], data=model_df, alpha=0.5)
    # we only want to plot the fixed effects line
    fixed_vals = mdf.fe_params['Intercept'] + mdf.fe_params['log_age'] * model_df['log_age']
    sns.lineplot(x=model_df['days_since_published'], y=fixed_vals, color='red')
    plt.title(f'Log {col} vs. days with MixedLM Regression Line')
    plt.show()

# %%
df.head()
# %%
# finally
# add the maturity rating as a numeric code
rating_map = {
    'General Audiences': 0,
    'Teen And Up Audiences': 1,
    'Mature': 2,
    'Explicit': 3,
    'Not Rated': np.nan} # so higher number means more mature content
df['maturity_rating'] = df['rating'].map(rating_map)

df.head()


# %%
# save to json
df.to_json(f"../data/{ts}_fanfics_metadata_with_residuals.json", orient='records', lines=True)
# %%
