# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import statsmodels.api as sm
import statsmodels.formula.api as smf
import time
from tqdm import tqdm
from pathlib import Path
import ast


# timestamp
ts = time.strftime("%Y-%m-%d")
print("Timestamp:", ts)

CWD = Path.cwd()
DATA_PATH = CWD.parent / "data"
FANFIC_DATA = DATA_PATH / "MythFic_txt"

# %%

## process fanfic corpus ##

# clean corpus
meta = pd.read_csv(FANFIC_DATA / 'fanfics_Greek_myth_metadata.csv')
print(f'Initial metadata has {len(meta)} entries.')
# make ids strings
meta['work_id'] = meta['work_id'].astype(str)
# print length
print("org data has: ", len(meta))

# check which ids do not exist as text files
for id in meta['work_id']:
    if not Path.exists(FANFIC_DATA / f'{id}.txt'):
        print(f'Missing file for id: {id}')

# drop rows with missing text files
meta = meta[meta['work_id'] != "38183230"]
print(f'Cleaned metadata has {len(meta)} entries.')
meta.head()
# %%
# texts are in mythfict_txt folder, txt files named by their 'id' in metadata

texts = []
for fid in tqdm(meta['work_id']):
    with open(FANFIC_DATA / f'{fid}.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        texts.append(text)
meta['text'] = texts
meta.head()

# %%

# fix some cols
df = meta.copy()

# AUTHOR
# 1. Parse string representations of lists into actual lists
df['author'] = df['author'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [x] if isinstance(x, str) else x)
# 2. Extract authors (now that we have real lists)
df['author'] = df['author'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

# see nans in author_1
nan_counts_authors = df['author'].isna().sum()
print("Number of NaNs in author column:", nan_counts_authors)

# DATE
df['published'] = pd.to_datetime(df['published'], errors='coerce')
# add time since published column
reference_date = pd.to_datetime("2023-01-01")
diff = reference_date - df['published']
df['days_since_published'] = diff.dt.days

rating_map = {
    'General Audiences': 0,
    'Teen And Up Audiences': 1,
    'Mature': 2,
    'Explicit': 3,
    'Not Rated': np.nan} # so higher number means more mature content
df['maturity_rating'] = df['rating'].map(rating_map)

print("Original length of dataframe:", len(df))
df.head()
# %%

##### engagement metrics #####

# reception metrics
engagement_cols = ['kudos', 'comments', 'hits', 'days_since_published']

# check nan in comments and hits
reception_cols = ['comments', 'hits', 'kudos']

for col in reception_cols:
    print(f"Number of NaNs in {col}: {df[col].isna().sum()}")
    # fillna with 0
    df[col] = df[col].fillna(0)


# %%

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

fig, axes = plt.subplots(3, 2, figsize=(15, 9))
axes = axes.flatten()
for i, col in enumerate(engagement_cols + ['kudos_hits_ratio', 'comment_hits_ratio']):
    # shapiro
    stat, p = shapiro(tmp[col].dropna())
    print(f'Shapiro-Wilk test for {col}: stat={stat:.4f}, p={p:.4f}')
    ax = axes[i]
    sns.histplot(df[col], kde=True, ax=ax)
    # rename x-axis
    ax.set_xlabel(col.replace('_', ' ').title())
sns.set_style("whitegrid")
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

    # first print the spearman corr with days_since_published
    corr = df[[col, 'days_since_published']].dropna().corr(method='spearman').iloc[0,1]
    print(f"Spearman correlation of {col} with days_since_published: {corr:.4f}")

    # # add spearman correlation of residuals with days_since_published
    # print("\n")
    # print(f"spearman of residuals {col} with days_since_published:")
    # corr = df[[f'{col}_resid', 'days_since_published']].dropna().corr(method='spearman').iloc[0,1]
    # print(f"{corr:.4f}")

    # and lets plot the regression
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
def clean_author(x):
    if isinstance(x, list):
        return x[0] if len(x) > 0 else None  # Empty list → None
    return x

df['author'] = df['author'].apply(clean_author)
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

print(df.columns)
df.head()
# %%
# save to json
df.to_json(DATA_PATH / f"{ts}_fanfics_cleaned.json", orient='records', lines=True)
# %%
