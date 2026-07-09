
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, ttest_ind, mannwhitneyu
import time
import numpy as np
from pathlib import Path

# make time stamp
ts = time.strftime("%Y-%m-%d_%H-%M")
print(f"Timestamp: {ts}")

CWD = Path(__file__).parent
DATA_PATH = CWD.parent / "data"
FIGS = CWD.parent / "figs"

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

path = DATA_PATH / "2025-11-19_fanfics_metadata_with_residuals.json"
fanfic = pd.read_json(path, orient='records', lines=True)
print(fanfic.columns)
print(f"len of fanfic: {len(fanfic)}")
fanfic.head()

# %%
gendata = DATA_PATH / "scored_data/simplestories_lemmatized_with_scores.json"
gen_df = pd.read_json(gendata, orient='records', lines=True)
print(gen_df.columns)
gen_df.head()

# %%

# get Chicago
chic = pd.read_csv(DATA_PATH / "chicago_sensory_profiles_by_file.csv")
print(chic.columns)
# define our sense columns
chic_sense_cols = [x for x in chic.columns if x.endswith('_mean') and any(sense in x for sense in ['auditory', 'gustatory', 'haptic', 'interoceptive', 'olfactory', 'visual'])]
chic = chic[chic_sense_cols + ['file_id']].copy()

# meta_chic
meta = pd.read_excel("/Users/au324704/Desktop/CHICAGO_MEASURES_MARCH24.xlsx")
meta = meta[["BOOK_ID", "AUTH_FIRST", "AUTH_LAST", "WORDCOUNT"]]
meta.columns = ["file_id", "author_first", "author_last", "text_length"]
meta["author"] = meta["author_first"] + " " + meta["author_last"]
meta.drop(columns=["author_first", "author_last"], inplace=True)
chic = chic.merge(meta, how='left', on='file_id')
# rename cols
chic.rename(columns={col: "avg_matched_" + col.replace("_mean", ".mean") for col in chic.columns if col.endswith('_mean') and any(sense in col for sense in ['auditory', 'gustatory', 'haptic', 'interoceptive', 'olfactory', 'visual'])}, inplace=True)
chic.head()


# %%

##### sense columns and derived metrics #####

# decide which columns to use, here senses + normalized & engagement metrics
sense_cols = ["auditory.mean", "gustatory.mean", "olfactory.mean", "haptic.mean", "visual.mean", "interoceptive.mean"]
use_what = "avg_matched_" # set this to total, avg_matched, or normalized
sense_cols_prefixed = [use_what + col for col in sense_cols]

datasets = {"fanfic": fanfic, "chicago": chic, "simplestories": gen_df}

# for each dataset, we want to add entropy
def add_entropy(df, sense_cols_prefixed):
    df[f'{use_what}sense_sum'] = df[sense_cols_prefixed].sum(axis=1)
    # add a column for the percent of each sense
    for sense in sense_cols_prefixed:
        colname = sense.replace('.mean', '_percent')
        df[f"{colname}"] = df[sense] / df[f'{use_what}sense_sum']
    # entropy of the sense distribution (we use the percent columns for this, since they sum to 1)
    df['sense_entropy'] = df[[col for col in df.columns if col.endswith('_percent')]].apply(lambda x: entropy(x, base=2), axis=1)
    return df

# apply
for name, df in datasets.items():
    datasets[name] = add_entropy(df, sense_cols_prefixed)

# define additional sense columns to use
add_sense_cols = [f'{use_what}sense_sum', 'sense_entropy'] 
percent_sense_cols = [col for col in df.columns if col.endswith('_percent')] # percent show some of the same info so we skip them for now

# %%

for df in datasets.keys():
    print(f"Dataset: {df}")
    for sense in sense_cols_prefixed:
        print(f"{sense}: {datasets[df][sense].mean():.3f}", f"{datasets[df][sense].std():.3f}")
    print("==========")


# %%

# we want to do a kde plot of the distribution of each sense score for fanfiction and chicago, to see how they compare
senses = ["auditory", "gustatory", "olfactory", "haptic", "visual", "interoceptive"]

plt.figure(figsize=(18, 3))
sns.set_style("whitegrid")

for i, sense in enumerate(senses):
    plt.subplot(1, len(senses), i + 1)
    for name, df in datasets.items():
        sns.kdeplot(data=df, x=f'{use_what}{sense}.mean', label=name.capitalize(), fill=True, alpha=0.2)
    plt.xlabel(f'{sense.capitalize()}')
    if i == 0:
        plt.legend(loc='upper left')
        plt.ylabel('Density')
    else:
        plt.ylabel('')
plt.tight_layout()
plt.savefig(f"../figs/{ts}_sense_score_distributions_based_on_{use_what}.png", bbox_inches='tight')
plt.show()

# %%

# we also want to see how the values distribute across the senses for each dataset, so we want to do a histplot of the percent for each sense, faceted by dataset
# Calculate average percentages
avg_senses = {}
for name, df in datasets.items():
    sense_cols = [f'{use_what}{sense}_percent' for sense in senses]
    avg_senses[name] = df[sense_cols].mean()

avg_df = pd.DataFrame(avg_senses).T  # Datasets as rows, senses as columns

plt.figure(figsize=(8, 6))
avg_df.plot(kind='bar', stacked=True, ax=plt.gca(), color=sns.color_palette("husl", len(senses)))
plt.ylabel('Average Percentage')
plt.xlabel('Dataset')
plt.title('Average Sense Distribution by Dataset')
# set the legend to the senses list
plt.legend(title='Sense', labels=[sense.capitalize() for sense in senses], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


from math import pi

# Calculate averages
avg_df = pd.DataFrame({name: df[[f'{use_what}{s}_percent' for s in senses]].mean()
                      for name, df in datasets.items()}).T

# Plot
plt.figure(figsize=(6, 6))
angles = [n / float(len(senses)) * 2 * pi for n in range(len(senses))]
angles += angles[:1]

ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], senses)

for name, row in avg_df.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=name.capitalize())
    ax.fill(angles, values, alpha=0.25)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.7))
plt.show()

# %%
gen_df[["generation_id","persona","style","theme","topic"]]

gen_df["author"] = gen_df["persona"] + "_" + gen_df["style"]

# print all authors > 20 VC
print("Authors with more than 20 generated stories:")
print(gen_df["author"].value_counts()[gen_df["author"].value_counts() > 10])

# %%
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from sklearn.preprocessing import label_binarize


fanfic["label"] = 1
chic["label"] = 0
gen_df["label"] = 2

# add text length
fanfic["text_length"] = fanfic["text"].apply(lambda x: len(x.split()))
gen_df["text_length"] = gen_df["text"].apply(lambda x: len(x.split()))

# downsample chicago
chic_sampled = chic.sample(n=fanfic.shape[0], random_state=42)
gen_df_sampled = gen_df.sample(n=fanfic.shape[0], random_state=42)

cols = sense_cols_prefixed + ["label", "author", "text_length", "sense_entropy"]

together = pd.concat([fanfic[cols], chic_sampled[cols], gen_df_sampled[cols]], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# create unique dummy author IDs for missing authors
missing_mask = together["author"].isna()
together.loc[missing_mask, "author"] = [f"missing_author_{i}" for i in range(missing_mask.sum())]
groups = together["author"].astype(str)
print(f"Balanced dataset stats: {together['label'].value_counts()}")
print(f"Number of unique authors: {groups.nunique()}")
print(f"N unique authors per label: {together.groupby('label')['author'].nunique()}")
print(f"Avg number of samples per author: {together.groupby('author').size().mean():.2f}")
print(f"Avg textlength per label: {together.groupby('label')['text_length'].mean()}")

# now drop text_length column since we don't need it for classification
together = together.drop(columns=["text_length"])


together.head(20)
# %%
X = together[sense_cols_prefixed]
y = together["label"]
groups = together["author"]


# Setup: Use multinomial logistic regression for 3 classes
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        multi_class='multinomial',  # Critical for 3+ classes
        solver='lbfgs',              # Supports multinomial
        max_iter=1000,
        random_state=42))

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

scores = {
    "fold": [],
    "accuracy": [],
    "macro_precision": [],
    "macro_recall": [],
    "macro_f1": [],
    "roc_auc_ovr": []} # One-vs-Rest AUC for multi-class

per_class_scores = {
    '0_precision': [], '0_recall': [], '0_f1': [],
    '1_precision': [], '1_recall': [], '1_f1': [],
    '2_precision': [], '2_recall': [], '2_f1': []}


all_coefs = []  # Store coefficients for ALL classes

# save all true and predicted labels for confusion matrix
all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # Shape: (n_samples, 3)

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    # Multi-class ROC AUC (One-vs-Rest)
    roc_auc = roc_auc_score(
        label_binarize(y_test, classes=[0, 1, 2]),
        y_proba,
        multi_class='ovr'
    )

    # Use macro-average for multi-class metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    for class_label in ['0', '1', '2']:
        per_class_scores[f'{class_label}_precision'].append(report[class_label]['precision'])
        per_class_scores[f'{class_label}_recall'].append(report[class_label]['recall'])
        per_class_scores[f'{class_label}_f1'].append(report[class_label]['f1-score'])

    scores["fold"].append(fold)
    scores["accuracy"].append(model.score(X_test, y_test))
    scores["macro_precision"].append(report['macro avg']['precision'])
    scores["macro_recall"].append(report['macro avg']['recall'])
    scores["macro_f1"].append(report['macro avg']['f1-score'])
    scores["roc_auc_ovr"].append(roc_auc)

    all_coefs.append(model.named_steps['logisticregression'].coef_)

# Average scores
print("\n--- Average Performance Across Folds ---")
for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]:
    print(f"{metric}: {np.mean(scores[metric]):.4f} ± {np.std(scores[metric]):.4f}")

# Print per-class results
print("\n--- Per-Class Performance (mean ± std) ---")
for class_label in ['0', '1', '2']:
    print(f"\nClass {class_label}:")
    for metric in ['precision', 'recall', 'f1']:
        key = f'{class_label}_{metric}'
        print(f"  {metric}: {np.mean(per_class_scores[key]):.4f} ± {np.std(per_class_scores[key]):.4f}")

# Coefficients: average across folds for each class
coef_array = np.array(all_coefs)  # Shape: (n_folds, n_classes, n_features)
for class_idx in range(coef_array.shape[1]):  # Use shape[1] instead of hardcoded 3
    print(f"\nClass {class_idx} coefficients (mean ± std):")
    print(pd.DataFrame({
        'mean': coef_array[:, class_idx, :].mean(axis=0),
        'std': coef_array[:, class_idx, :].std(axis=0)}, index=sense_cols_prefixed).sort_values('mean', ascending=False))


# After the loop, create and plot the matrix:
cm = confusion_matrix(all_y_true, all_y_pred)

plt.figure(figsize=(3, 3), dpi=500)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Published', 'Fanfiction', 'SimStories'],
            yticklabels=['Published', 'Fanfiction', 'SimStories'])
plt.xlabel(r'$\bf{Predicted}$')
plt.ylabel(r'$\bf{True}$')
plt.xticks(rotation=10)
plt.yticks(rotation=0)
plt.show()


# %%
# get the sense entropy correlation per sense for each dataset

for dataset in datasets.keys():
    print(f"Dataset: {dataset}")
    df_corr = datasets[dataset][sense_cols_prefixed + ["sense_entropy"]].corr(method='spearman')
    plt.figure(figsize=(6, 6))
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, square=True)
    plt.title(f'Correlation Heatmap between Sensorimotor Scores and Sense Entropy for {dataset.capitalize()}')
    plt.show()
    print("==========")


# %%

classes = ['Published', 'Fanfiction', 'SimpleStories']
colors = ['#D81B60', '#1E88E5', '#E1BE6A']

fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True, dpi=500)
for ax, i, cls, color in zip(axes, range(3), classes, colors):
    means = coef_array[:, i, :].mean(axis=0)
    stds = coef_array[:, i, :].std(axis=0)
    y = np.arange(len(senses))
    y_display = np.arange(len(senses))[::-1]  # reverses display order only


    # # individual folds
    # for fold in range(coef_array.shape[0]):
    #     jitter = np.random.normal(0, 0.15, len(y))
    #     ax.scatter(coef_array[fold, i, :], y + jitter, color='black', alpha=1, s=5, zorder=1)

    # mean markers: de-emphasize if |mean| < 1*std (i.e. consistent with zero)
    reliable = np.abs(means) >= stds
    
    # reliable means: full color, full opacity
    ax.scatter(
        means[reliable], y_display[reliable],
        color=color, edgecolor='black', linewidth=1,
        s=80, alpha=1, zorder=2,
    )
    # unreliable means: gray, faded, smaller
    ax.scatter(
        means[~reliable], y_display[~reliable],
        color='white', edgecolor='black', linewidth=1,
        s=80, zorder=2,
    )

    # error bars: ±1 SD, drawn on top so whiskers are never hidden
    ax.errorbar(
        means, y_display, xerr=stds,
        fmt='none', ecolor='black', elinewidth=.5,
        capsize=13, capthick=.5, zorder=1,
    )

    ax.axvline(0, color='0.1', linestyle='--', zorder=0.5)
    ax.set_title(cls)
    ax.set_xlim(-6.5, 7)

    ax.grid(True, color='0.4', linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

axes[0].set_yticks(range(len(senses)))
axes[0].set_yticklabels([x.title() for x in senses[::-1]], fontsize=12)
fig.supxlabel("Feature coefficient (log-odds)")
plt.tight_layout()
plt.show()

# %%







resid_cols = [x for x in df.columns if x.endswith('_resid')]

# formalize the columns we want to look at
engagement_cols = ['days_since_published', 'kudos_hits_ratio', 'comment_hits_ratio', 'kudos', 'comments', 'hits', 'maturity_rating']

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

# %%




# BRING IN ANOTHER DATASET TO SEE WHETHER CORRELATIONS WITH SENS ENYTROPY HOLD UP


# ALSO ADD GENERATED DATA THANKS

# %%


# compute the skewness metric of the sense distribution per book
# why did we want to do this? ah we wanted to see if there is a "visual" style vs other styles, and whether that correlates with engagement metrics. we can do this by looking at the skewness of the sense distribution, or by looking at the percent of the dominant sense. let's do both.
from scipy.stats import skew, kurtosis





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
