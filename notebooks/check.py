
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import time
import numpy as np
from pathlib import Path
import openpyxl
import re
from math import pi


# make time stamp
ts = time.strftime("%Y-%m-%d_%H-%M")
print(f"Timestamp: {ts}")

CWD = Path(__file__).parent
DATA_PATH = CWD.parent / "data" / "lemmatized_data" / "scored_data"
FIGS = CWD.parent / "figs"
OUT_DIR = CWD / "OUT_DIR"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%

datasets = {}

for path in Path(DATA_PATH).glob("*.json"):
    name = path.stem.replace("_with_scores", "").replace("_lemmatized", "")
    datasets[name] = pd.read_json(path, orient='records', lines=True)
    print(f"Loaded {name} dataset with {len(datasets[name])} entries.")

datasets["fanfics"].head()


datasets.keys()

# %%

storyscope = datasets["storyscope"]

# extract author suffix from work_id (everything after the last underscore)
storyscope["author"] = storyscope["work_id"].str.extract(r'_([a-zA-Z]+)$')

print(storyscope["author"].value_counts())

# split into separate datasets, keyed like storyscope_gpt, storyscope_claude, etc.
for author, group in storyscope.groupby("author"):
    key = f"storyscope_{author}"
    datasets[key] = group.reset_index(drop=True)
    print(f"Loaded {key} dataset with {len(datasets[key])} entries.")

del datasets["storyscope"]
datasets.keys()

# %%

# # get Chicago
# chic = pd.read_csv(DATA_PATH / "chicago_sensory_profiles_by_file.csv")
# print(chic.columns)
# # define our sense columns
# chic_sense_cols = [x for x in chic.columns if x.endswith('_mean') and any(sense in x for sense in ['auditory', 'gustatory', 'haptic', 'interoceptive', 'olfactory', 'visual'])]
# chic = chic[chic_sense_cols + ['file_id']].copy()

# meta_chic
meta = pd.read_excel(CWD.parent / "data" / "CHICAGO_MEASURES_MARCH24.xlsx")
meta = meta[["BOOK_ID", "AUTH_FIRST", "AUTH_LAST", "WORDCOUNT", "PUBL_DATE", "LIBRARIES", "RATING_COUNT", ]]
meta.columns = ["work_id", "author_first", "author_last", "text_length", "year", "libraries", "rating_count"]
meta["author"] = meta["author_first"] + " " + meta["author_last"]
meta.drop(columns=["author_first", "author_last"], inplace=True)
chic = datasets["chicago"].merge(meta, how='left', on='work_id')
# rename cols
chic.rename(columns={col: "avg_matched_" + col.replace("_mean", ".mean") for col in chic.columns if col.endswith('_mean') and any(sense in col for sense in ['auditory', 'gustatory', 'haptic', 'interoceptive', 'olfactory', 'visual'])}, inplace=True)
chic.head()


# %%
datasets["chicago"] = chic
datasets.keys()

# %%
# check before filtering
print(f"Fanfics before filtering: {len(datasets['fanfics'])}")

# drop rows with zero total sense score (degenerate/near-empty texts)
zero_mask = datasets["fanfics"][sense_cols_prefixed].sum(axis=1) == 0
datasets["fanfics"] = datasets["fanfics"][~zero_mask].reset_index(drop=True)

print(f"Fanfics after filtering: {len(datasets['fanfics'])} (dropped {zero_mask.sum()})")

# %%

##### sense columns and derived metrics #####

# decide which columns to use, here senses + normalized & engagement metrics
sense_cols = ["auditory.mean", "gustatory.mean", "olfactory.mean", "haptic.mean", "visual.mean", "interoceptive.mean"]
use_what = "avg_matched_" # set this to total, avg_matched, or normalized
sense_cols_prefixed = [use_what + col for col in sense_cols]


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

output_path = OUT_DIR / f"{ts}_sense_score_summary.txt"

with open(output_path, "w") as f:
    for name in datasets.keys():
        f.write(f"Dataset: {name}\n")
        print(f"Dataset: {name}")
        for sense in sense_cols_prefixed:
            line = f"{sense}: {datasets[name][sense].mean():.3f} {datasets[name][sense].std():.3f}"
            f.write(line + "\n")
            print(line)
        entropy_line = f"sense_entropy: {datasets[name]['sense_entropy'].mean():.3f} {datasets[name]['sense_entropy'].std():.3f}"
        f.write(entropy_line + "\n")
        print(entropy_line)
        f.write("==========\n")
        print("==========")

print(f"\nSaved summary to {output_path}")

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
plt.savefig(FIGS / f"{ts}_sense_score_distributions_based_on_{use_what}.png", bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
axes = axes.flatten()

emphasize = {"chicago", "fanfics", "simplestories"}

for i, sense in enumerate(senses):
    ax = axes[i]
    for name, df in datasets.items():
        is_main = name in emphasize
        sns.kdeplot(
            data=df, x=f'{use_what}{sense}.mean', label=name.capitalize(),
            fill=is_main, alpha=0.25 if is_main else 0.9,
            linewidth=2 if is_main else 1,
            ax=ax
        )
    ax.set_xlabel(sense.capitalize())
    ax.set_ylabel('Density' if i % 3 == 0 else '')

handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    if ax.get_legend(): ax.get_legend().remove()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)

plt.tight_layout()
plt.savefig(FIGS / f"{ts}_sense_score_distributions_based_on_{use_what}.png", bbox_inches='tight')
plt.show()

# %%

from math import pi

# collapse Storyscope models into one group, keep everything else separate
main_groups = {
    "Chicago": datasets["chicago"],
    "Fanfics": datasets["fanfics"],
    "SimpleStories": datasets["simplestories"],
    "Storyscope": pd.concat(
        [df for name, df in datasets.items() if name.startswith("storyscope_")]
    ),
}

percent_cols = [f'{use_what}{s}_percent' for s in senses]
avg_df = pd.DataFrame({name: df[percent_cols].mean() for name, df in main_groups.items()}).T
avg_df.columns = senses  # clean labels

angles = [n / float(len(senses)) * 2 * pi for n in range(len(senses))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], [s.capitalize() for s in senses], fontsize=11)
ax.tick_params(axis='x', pad=15)

ax.set_rlabel_position(0)
plt.yticks(fontsize=8, color='black')
ax.set_ylim(0, avg_df.values.max() * 1.15)

colors = sns.color_palette("colorblind", len(avg_df))
markers = ['o', 'o','o', 'o', 'o']

for (name, row), color, marker in zip(avg_df.iterrows(), colors, markers):
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, marker=marker, markersize=6,
             label=name, color=color)

plt.legend(loc='upper right', bbox_to_anchor=(0.9, 1), frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(f"../figs/{ts}_radar_modalities.png", bbox_inches='tight', dpi=500)
plt.show()

# %%
datasets.keys()

# %%

datasets['simplestories']["author"] = datasets['simplestories']["persona"] + "_" + datasets['simplestories']["style"]

# print all authors > 20 VC
print("Authors with more than 20 generated stories:")
print(datasets['simplestories']["author"].value_counts()[datasets['simplestories']["author"].value_counts() > 10])

for gen_df in ['storyscope_claude', 'storyscope_deepseek', 'storyscope_gemini', 'storyscope_gpt', 'storyscope_kimi']:
    datasets[gen_df]['author'] = datasets[gen_df]['author'] + "_" + datasets[gen_df]['human_author']
    print("Authors with more than 20 generated stories:")
    print(datasets[gen_df]["author"].value_counts()[datasets[gen_df]["author"].value_counts() > 10])




# %%

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import pyarrow as pa
print(pd.__version__)
print(pa.__version__)

# --- assign integer labels to every dataset key, in a fixed, reproducible order ---
class_names = sorted(datasets.keys())          # fixed order so label ints are stable across runs
label_map = {name: i for i, name in enumerate(class_names)}
print("Label mapping:", label_map)

cols = sense_cols_prefixed + ["label", "author", "n_tokens", "sense_entropy"]

# smallest dataset sets the downsample target for balanced classes
min_n = min(len(df) for df in datasets.values())
print(f"Downsampling every class to n={min_n} (smallest: "
      f"{min(datasets, key=lambda k: len(datasets[k]))})")

pieces = []
for name, df in datasets.items():
    d = df.copy()
    d["label"] = label_map[name]
    d = d.sample(n=min_n, random_state=42)
    pieces.append(d[cols])

together = pd.concat(pieces, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# create unique dummy author IDs for missing authors
missing_mask = together["author"].isna()
together.loc[missing_mask, "author"] = [f"missing_author_{i}" for i in range(missing_mask.sum())]
groups = together["author"].astype(str)

print(f"Balanced dataset stats:\n{together['label'].value_counts().sort_index()}")
print(f"Number of unique authors: {groups.nunique()}")
print(f"N unique authors per label: {together.groupby('label')['author'].nunique()}")
print(f"Avg number of samples per author: {together.groupby('author').size().mean():.2f}")
print(f"Avg textlength per label: {together.groupby('label')['n_tokens'].mean()}")

# drop text_length, not used as a feature (but keep it in a saved file if you want the length-confound check later)
together = together.drop(columns=["n_tokens"])

together["author"] = together["author"].astype("object")

out_path = OUT_DIR / f"{ts}_classification_dataset.csv.gz"

together.to_csv(out_path, index=False, compression="gzip")

print(f"Saved to {out_path}")
print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

# %%

# %%

X = together[sense_cols_prefixed]
y = together["label"]
groups = together["author"]

# derive class info from the data instead of hardcoding
class_labels = sorted(y.unique())  # e.g. [0, 1, 2, 3, ...]
n_classes = len(class_labels)

# human-readable names for plotting — inverts the label_map you built earlier
# (label_map = {name: i for i, name in enumerate(class_names)})
inv_label_map = {v: k for k, v in label_map.items()}
class_display_names = [inv_label_map[c].replace("_", " ").title() for c in class_labels]

print(f"Classes found: {dict(zip(class_labels, class_display_names))}")

# Setup: multinomial logistic regression, works for any n_classes >= 2
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42))

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

scores = {
    "fold": [],
    "accuracy": [],
    "macro_precision": [],
    "macro_recall": [],
    "macro_f1": [],
    "roc_auc_ovr": []}

# build per-class score dict dynamically
per_class_scores = {f'{c}_{metric}': [] for c in class_labels for metric in ['precision', 'recall', 'f1']}

all_coefs = []
all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # Shape: (n_samples, n_classes)

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    # Multi-class ROC AUC (One-vs-Rest) — works for any n_classes
    roc_auc = roc_auc_score(
        label_binarize(y_test, classes=class_labels),
        y_proba,
        multi_class='ovr'
    )

    report = classification_report(y_test, y_pred, output_dict=True)

    for c in class_labels:
        key = str(c)  # classification_report keys are strings of the label
        per_class_scores[f'{c}_precision'].append(report[key]['precision'])
        per_class_scores[f'{c}_recall'].append(report[key]['recall'])
        per_class_scores[f'{c}_f1'].append(report[key]['f1-score'])

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

# Per-class results, using real names
print("\n--- Per-Class Performance (mean ± std) ---")
for c, name in zip(class_labels, class_display_names):
    print(f"\nClass {c} ({name}):")
    for metric in ['precision', 'recall', 'f1']:
        key = f'{c}_{metric}'
        print(f"  {metric}: {np.mean(per_class_scores[key]):.4f} ± {np.std(per_class_scores[key]):.4f}")

# Coefficients: average across folds for each class
coef_array = np.array(all_coefs)  # Shape: (n_folds, n_classes, n_features)
for class_idx in range(coef_array.shape[1]):
    name = class_display_names[class_idx]
    print(f"\nClass {class_labels[class_idx]} ({name}) coefficients (mean ± std):")
    print(pd.DataFrame({
        'mean': coef_array[:, class_idx, :].mean(axis=0),
        'std': coef_array[:, class_idx, :].std(axis=0)}, index=sense_cols_prefixed).sort_values('mean', ascending=False))

# Confusion matrix, sized and labeled dynamically
cm = confusion_matrix(all_y_true, all_y_pred, labels=class_labels)

fig_size = max(3, n_classes * 0.9)  # scale figure size with number of classes
plt.figure(figsize=(fig_size, fig_size), dpi=500)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_display_names,
            yticklabels=class_display_names)
plt.xlabel(r'$\bf{Predicted}$')
plt.ylabel(r'$\bf{True}$')
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %%

# --- build class groups: Chicago, Fanfics, SimpleStories, Storyscope (combined) ---
storyscope_combined = pd.concat(
    [df for name, df in datasets.items() if name.startswith("storyscope_")],
    axis=0
)

class_groups = {
    "chicago": datasets["chicago"],
    "fanfics": datasets["fanfics"],
    "simplestories": datasets["simplestories"],
    "storyscope": storyscope_combined,
}

class_names = sorted(class_groups.keys())
label_map = {name: i for i, name in enumerate(class_names)}
print("Label mapping:", label_map)

cols = sense_cols_prefixed + ["label", "author", "n_tokens", "sense_entropy"]

# downsample every class to Fanfiction's size, as before
target_n = len(class_groups["fanfics"])
print(f"Downsampling every class to n={target_n} (Fanfiction size)")

pieces = []
for name, df in class_groups.items():
    d = df.copy()
    d["label"] = label_map[name]
    n = min(target_n, len(d))  # guard in case a class is smaller than target_n
    if n < target_n:
        print(f"WARNING: {name} has only {len(d)} rows, less than target {target_n}")
    d = d.sample(n=n, random_state=42)
    pieces.append(d[cols])

together = pd.concat(pieces, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# dummy author IDs for missing authors
missing_mask = together["author"].isna()
together.loc[missing_mask, "author"] = [f"missing_author_{i}" for i in range(missing_mask.sum())]
groups = together["author"].astype(str)

print(f"Balanced dataset stats:\n{together['label'].value_counts().sort_index()}")
print(f"Number of unique authors: {groups.nunique()}")
print(f"N unique authors per label: {together.groupby('label')['author'].nunique()}")
print(f"Avg number of samples per author: {together.groupby('author').size().mean():.2f}")
print(f"Avg textlength per label: {together.groupby('label')['n_tokens'].mean()}")

together = together.drop(columns=["n_tokens"])

X = together[sense_cols_prefixed]
y = together["label"]
groups = together["author"]

class_labels = sorted(y.unique())
n_classes = len(class_labels)

inv_label_map = {v: k for k, v in label_map.items()}
class_display_names = [inv_label_map[c].replace("_", " ").title() for c in class_labels]

print(f"Classes found: {dict(zip(class_labels, class_display_names))}")

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42))

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

scores = {
    "fold": [],
    "accuracy": [],
    "macro_precision": [],
    "macro_recall": [],
    "macro_f1": [],
    "roc_auc_ovr": []}

per_class_scores = {f'{c}_{metric}': [] for c in class_labels for metric in ['precision', 'recall', 'f1']}

all_coefs = []
all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    roc_auc = roc_auc_score(
        label_binarize(y_test, classes=class_labels),
        y_proba,
        multi_class='ovr'
    )

    report = classification_report(y_test, y_pred, output_dict=True)

    for c in class_labels:
        key = str(c)
        per_class_scores[f'{c}_precision'].append(report[key]['precision'])
        per_class_scores[f'{c}_recall'].append(report[key]['recall'])
        per_class_scores[f'{c}_f1'].append(report[key]['f1-score'])

    scores["fold"].append(fold)
    scores["accuracy"].append(model.score(X_test, y_test))
    scores["macro_precision"].append(report['macro avg']['precision'])
    scores["macro_recall"].append(report['macro avg']['recall'])
    scores["macro_f1"].append(report['macro avg']['f1-score'])
    scores["roc_auc_ovr"].append(roc_auc)

    all_coefs.append(model.named_steps['logisticregression'].coef_)

print("\n--- Average Performance Across Folds ---")
for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]:
    print(f"{metric}: {np.mean(scores[metric]):.4f} ± {np.std(scores[metric]):.4f}")

print("\n--- Per-Class Performance (mean ± std) ---")
for c, name in zip(class_labels, class_display_names):
    print(f"\nClass {c} ({name}):")
    for metric in ['precision', 'recall', 'f1']:
        key = f'{c}_{metric}'
        print(f"  {metric}: {np.mean(per_class_scores[key]):.4f} ± {np.std(per_class_scores[key]):.4f}")

coef_array = np.array(all_coefs)
for class_idx in range(coef_array.shape[1]):
    name = class_display_names[class_idx]
    print(f"\nClass {class_labels[class_idx]} ({name}) coefficients (mean ± std):")
    print(pd.DataFrame({
        'mean': coef_array[:, class_idx, :].mean(axis=0),
        'std': coef_array[:, class_idx, :].std(axis=0)}, index=sense_cols_prefixed).sort_values('mean', ascending=False))

cm = confusion_matrix(all_y_true, all_y_pred, labels=class_labels)

fig_size = max(3, n_classes * 0.9)
plt.figure(figsize=(fig_size, fig_size), dpi=500)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_display_names,
            yticklabels=class_display_names)
plt.xlabel(r'$\bf{Predicted}$')
plt.ylabel(r'$\bf{True}$')
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"../figs/{ts}_confusion_matrix_4class.png", bbox_inches='tight')
plt.show()

# %%






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
