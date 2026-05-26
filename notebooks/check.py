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
# df = pd.read_json("../data/fanfics_metadata_with_sensorimotor_scores.json", orient='records', lines=True)

# drop_cols = ['total_foot_leg.mean','normalized_foot_leg.mean', 'avg_matched_foot_leg.mean','total_hand_arm.mean', 'normalized_hand_arm.mean',
#        'avg_matched_hand_arm.mean', 'total_head.mean', 'normalized_head.mean','avg_matched_head.mean', 'total_mouth.mean', 'normalized_mouth.mean',
#        'avg_matched_mouth.mean', 'total_torso.mean', 'normalized_torso.mean','avg_matched_torso.mean']
# df = df.drop(columns=drop_cols)

# # add the sensitivity columns
# sensitivity_df = pd.read_json("../data/2025-11-18_12-39_fanfics_sensitivity_labelled.json", orient='records', lines=True)
# sensitivity_labels = [x for x in sensitivity_df.columns if x.startswith('sensitive_')] + ["work_id", "sensitivity_prop_above_threshold"] # just get the important columns
# sensitivity_df = sensitivity_df[sensitivity_labels]
# df = df.merge(sensitivity_df, how='left', on='work_id')
# df.head()

path = "../data/2025-11-19_fanfics_metadata_with_residuals.json"
df = pd.read_json(path, orient='records', lines=True)
df.head()
# %%

##### sense columns and derived metrics #####

# decide which columns to use, here senses + normalized & engagement metrics
sense_cols = ["auditory.mean", "gustatory.mean", "olfactory.mean", "haptic.mean", "visual.mean", "interoceptive.mean"]
use_what = "avg_matched_" # set this to total, avg_matched, or normalized
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

# print the avg sense scores for fanfiction
print("Average Sense Scores for Fanfiction:")
for sense in sense_cols_prefixed:
    print(f"{sense}: {df[sense].mean():.4f}")
    print(f"{df[sense].std():.4f}")

# %%
resid_cols = [x for x in df.columns if x.endswith('_resid')]

# formalize the columns we want to look at
engagement_cols = ['days_since_published', 'kudos_hits_ratio', 'comment_hits_ratio', 'hits', 'maturity_rating']

corr_cols = sense_cols_prefixed + add_sense_cols + engagement_cols + resid_cols # sensitive_cols + percent_sense_cols

# heatmap of correlations
plt.figure(figsize=(17, 15))
corr = df[corr_cols].corr(method='spearman')
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, square=True)
plt.title('Correlation Heatmap between Sensorimotor Scores and Engagement Metrics')
plt.savefig(f"../figs/{ts}_correlation_heatmap_based_on_{use_what}.png", bbox_inches='tight')
plt.show()

# heatmap of just the senses
plt.figure(figsize=(10, 8))
sense_corr = df[sense_cols_prefixed + add_sense_cols].corr(method='spearman')
sns.heatmap(sense_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, square=True)
plt.title('Correlation Heatmap between Sensorimotor Scores')
plt.savefig(f"../figs/{ts}_sense_correlation_heatmap_based_on_{use_what}.png", bbox_inches='tight')
plt.show()


# BRING IN ANOTHER DATASET TO SEE WHETHER CORRELATIONS WITH SENS ENYTROPY HOLD UP


# %%


# compute the skewness metric of the sense distribution per book




# %%

# plot the relationship between months since published and kudos_hits_ratio with regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x='days_since_published', y='kudos_hits_ratio_resid', data=df, alpha=0.5)
sns.lineplot(x='days_since_published', y='kudos_hits_ratio_resid', color='red', data=df)
plt.title('Kudos-Hits Ratio vs. Days Since Published with Regression Line')
plt.xlabel('Days Since Published')
plt.ylabel('Kudos-Hits Ratio')
plt.show()

# lets see the comment_hits_ratio vs months since published
plt.figure(figsize=(8, 6))
sns.scatterplot(x='days_since_published', y='comment_hits_ratio_resid', data=df, alpha=0.5)
plt.title('Comment-Hits Ratio vs. Days Since Published with Regression Line')
plt.xlabel('Days Since Published')
plt.ylabel('Comment-Hits Ratio')
plt.show()

# make historgrams of the residuals
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['kudos_hits_ratio_resid'].dropna(), kde=True)
plt.title('Histogram of Kudos-Hits Ratio Residuals')
plt.subplot(1, 2, 2)
sns.histplot(df['comment_hits_ratio_resid'].dropna(), kde=True)
plt.title('Histogram of Comment-Hits Ratio Residuals')
plt.show()
# %%
# plot the hits_resid across the maturity ratings
features_plot = ['kudos_hits_ratio_resid', 'comment_hits_ratio_resid', 'hits_resid']

plt.figure(figsize=(18, 5))
sns.set_style("whitegrid")

for i, feature in enumerate(features_plot):
    plt.subplot(1, 3, i + 1)
    sns.histplot(data=df, x=feature, hue='maturity_rating', element='step', alpha=0.2)
    plt.title(f'{feature}')
    plt.xlabel('Maturity Rating')
    plt.ylabel(feature)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"../figs/{ts}_residuals_across_maturity_ratings.png", bbox_inches='tight')
plt.show()


# %%

norms = pd.read_csv("../resources/cleaned_sensorimotor_norms.csv")
norms.head()

res = {}

for i, text in enumerate(df['lemmatized_text']):
    set_text = set(text)
    valid_tokens = [token for token in set_text if token in norms['word'].values]
    res[i] = {
        'num_tokens': len(text),
        'num_types': len(set_text),
        'num_valid_types': len(valid_tokens),
        'coverage': len(valid_tokens) / len(set_text) if set_text else 0}
    
# %%

res_df = pd.DataFrame.from_dict(res, orient='index')
res_df.head()

# %%
# merge with main df
df = df.merge(res_df, left_index=True, right_index=True)

# %%
# now let's correlate coverage with maturity_rating and engagement metrics
# see all columns correlated with coverage
rem_cols = ['work_id', 'title', 'author', 'rating', 'category', 'fandom',
       'relationship', 'character', 'additional tags', 'language', 'published',
       'status', 'status date', 'words', 'chapters', 'comments', 'kudos',
       'bookmarks', 'hits', 'text', 'lemmatized_text',]
norms_cols = [col for col in df.columns if any(sense in col for sense in ['auditory', 'gustatory', 'haptic', 'interoceptive', 'olfactory', 'visual'])]

df_corr = df.drop(columns=rem_cols + norms_cols).corr(method='spearman')
coverage_corr = df_corr.corr()['coverage'].sort_values(ascending=False)
print("Correlation of coverage with other columns:")
print(coverage_corr)

# show the distibution of the maturity ratings by coverage
mat = df[df['rating'] == 'Explicit']['coverage']
mat1 = df[df['rating'] == 'Mature']['coverage']
mat2 = df[df['rating'] == 'Teen And Up Audiences']['coverage']
mat3 = df[df['rating'] == 'General Audiences']['coverage']

dfs = [mat, mat1, mat2, mat3]

plt.figure(figsize=(10, 6))
for i, data in enumerate(dfs):
    sns.kdeplot(data, label=f'Maturity Rating {i+1}', fill=True, alpha=0.1)
plt.title('Distribution of Coverage by Maturity Rating')
plt.xlabel('Coverage')
plt.ylabel('Density')
plt.xlim(0.8, 1)
plt.legend(title='Maturity Rating', labels=['Explicit', 'Mature', 'Teen And Up Audiences', 'General Audiences', 'Not Rated'])
plt.show()

# %%
# linear model
# how much does maturity rating predict coverage
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('coverage ~ C(rating)', data=df).fit()
print(model.summary())

# %%
df
# %%
df['rating'].value_counts()
# %%
df.columns
# %%
# save to csv
df.to_csv(f"../data/{ts}_fanfics_metadata_with_sensorimotor_and_residuals_and_coverage.csv", index=False)
# %%
