
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

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# make time stamp
ts = time.strftime("%Y-%m-%d")
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
# finally delete the general one
del datasets["storyscope"]

### Make author-columns ####
# (i.e., for generated stuff, this looks like main style/prompt + model)

# simplestories
datasets['simplestories']["author"] = datasets['simplestories']["persona"] + "_" + datasets['simplestories']["style"]
# print all authors > 20 VC
print("Authors with more than 20 generated stories:")
print(datasets['simplestories']["author"].value_counts()[datasets['simplestories']["author"].value_counts() > 10])

# storyscope
for gen_df in ['storyscope_claude', 'storyscope_deepseek', 'storyscope_gemini', 'storyscope_gpt', 'storyscope_kimi']:
    datasets[gen_df]['author'] = datasets[gen_df]['author'] + "_" + datasets[gen_df]['human_author']
    print("Authors with more than 20 generated stories:")
    print(datasets[gen_df]["author"].value_counts()[datasets[gen_df]["author"].value_counts() > 10])

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

### REMOVE 0 ACROSS ALL FEATS ####

# check before filtering
# drop rows with zero total sense score (degenerate/near-empty texts)
for ds in list(datasets.keys()):
    zero_mask = datasets[ds][sense_cols_prefixed].sum(axis=1) == 0
    print(f"N zeroes {ds}: {zero_mask.sum()} out of {len(zero_mask)}")
    datasets[ds] = datasets[ds][~zero_mask].reset_index(drop=True)
    print(f"After filtering: {len(datasets[ds])} (dropped {zero_mask.sum()})")


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

### CLASSIFICATION ###

# ============================================================
# CONFIG
# ============================================================
COMBINE_STORYSCOPE = True   # True = one "storyscope" class; False = keep storyscope_gpt, storyscope_claude, etc. separate

random_state = 42

# ============================================================
# Build class groups
# ============================================================
if COMBINE_STORYSCOPE:
    storyscope_combined = pd.concat([df for name, df in datasets.items() if name.startswith("storyscope_")], axis=0)
    class_groups = {
        "chicago": datasets["chicago"],
        "fanfics": datasets["fanfics"],
        "simplestories": datasets["simplestories"],
        "storyscope": storyscope_combined }
    DATASETS_USED = "minimalist"
else:
    class_groups = {name: df for name, df in datasets.items() if name != "storyscope"}
    DATASETS_USED = "full"

class_names = sorted(class_groups.keys())
label_map = {name: i for i, name in enumerate(class_names)}
print("Label mapping:", label_map)

# %%
# ============================================================
# CONFIG: pool ALL generated sources into one class
# ============================================================
generated_combined = pd.concat(
    [datasets["simplestories"]] + [df for name, df in datasets.items() if name.startswith("storyscope_")],
    axis=0)

class_groups = {
    "chicago": datasets["chicago"],
    "fanfics": datasets["fanfics"],
    "generated": generated_combined,
}
DATASETS_USED = "three_regime"

class_names = sorted(class_groups.keys())
label_map = {name: i for i, name in enumerate(class_names)}
print("Label mapping:", label_map)

# ============================================================
# Assemble balanced dataset
# ============================================================
cols = sense_cols_prefixed + ["label", "author", "n_tokens", "sense_entropy"]

target_n = min(len(df) for df in class_groups.values())
print(f"Downsampling every class to n={target_n} (i.e.: {min(class_groups, key=lambda k: len(class_groups[k]))})")

pieces = []
for name, df in class_groups.items():
    d = df.copy()
    d["label"] = label_map[name]
    n = min(target_n, len(d))
    if n < target_n:
        print(f"WARNING: {name} has only {len(d)} rows, less than target {target_n}")
    d = d.sample(n=n, random_state=random_state)
    pieces.append(d[cols])

together = pd.concat(pieces, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

# give missing authors dummy-ids
missing_mask = together["author"].isna()
print(f"N isna author: {missing_mask.sum()}")
together.loc[missing_mask, "author"] = [f"missing_author_{i}" for i in range(missing_mask.sum())]
groups = together["author"].astype(str)

# LOG IT
output_path = OUT_DIR / f"{ts}_{DATASETS_USED}_classification_data_summary.txt"
with open(output_path, "w") as f:
    f.write(f"Balanced dataset stats:\n{together['label'].value_counts().sort_index()}\n")
    f.write(f"Number of unique authors: {groups.nunique()}\n")
    f.write(f"N unique authors per label: {together.groupby('label')['author'].nunique()}\n")
    f.write(f"Avg number of samples per author: {together.groupby('author').size().mean():.2f}\n")
    f.write(f"Avg textlength per label: {together.groupby('label')['n_tokens'].mean()}\n")
    f.write("==========\n")
print(f"Balanced dataset stats:\n{together['label'].value_counts().sort_index()}")
print(f"Number of unique authors: {groups.nunique()}")
print(f"N unique authors per label: {together.groupby('label')['author'].nunique()}")
print(f"Avg number of samples per author: {together.groupby('author').size().mean():.2f}")
print(f"Avg textlength per label: {together.groupby('label')['n_tokens'].mean()}")

together = together.drop(columns=["n_tokens"])

together["author"] = together["author"].astype("object")

out_path = OUT_DIR / f"{DATASETS_USED}_classification_dataset.csv.gz"
together.to_csv(out_path, index=False, compression="gzip")
print(f"Saved to {out_path}")


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
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGS / f"{ts}_{DATASETS_USED}_confusion_matrix.png")
plt.show()


import json

# ============================================================
# Structured output: per-fold metrics, per-class metrics, coefficients
# ============================================================
class_labels = sorted(int(c) for c in y.unique())  # plain Python ints, not numpy int64

results = {
    "timestamp": ts,
    "dataset_config": DATASETS_USED,
    "n_classes": n_classes,
    "class_labels": class_labels,
    "class_display_names": class_display_names,
    "label_map": label_map,
    "n_folds": len(scores["fold"]),

    "overall_metrics": {
        metric: {"mean": float(np.mean(scores[metric])), "std": float(np.std(scores[metric]))}
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]
    },

    "per_fold_metrics": {
        metric: [float(v) for v in scores[metric]]
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]
    },

    "per_class_metrics": {
        f"{name}": {
            metric: {
                "mean": float(np.mean(per_class_scores[f'{c}_{metric}'])),
                "std": float(np.std(per_class_scores[f'{c}_{metric}']))
            }
            for metric in ["precision", "recall", "f1"]
        }
        for c, name in zip(class_labels, class_display_names)
    },

    "coefficients": {
        name: {
            feature: {
                "mean": float(coef_array[:, class_idx, feat_idx].mean()),
                "std": float(coef_array[:, class_idx, feat_idx].std())
            }
            for feat_idx, feature in enumerate(sense_cols_prefixed)
        }
        for class_idx, name in enumerate(class_display_names)
    },

    "confusion_matrix": cm.tolist(),
}

results_path = OUT_DIR / f"{ts}_{DATASETS_USED}_classification_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved structured results to {results_path}")

# ============================================================
# Also save a tidy long-format CSV — one row per (class, metric) — easy for tables/plots later
# ============================================================

rows = []
for c, name in zip(class_labels, class_display_names):
    for metric in ["precision", "recall", "f1"]:
        rows.append({
            "class_label": c,
            "class_name": name,
            "metric": metric,
            "mean": np.mean(per_class_scores[f'{c}_{metric}']),
            "std": np.std(per_class_scores[f'{c}_{metric}']),
        })

per_class_df = pd.DataFrame(rows)
per_class_csv_path = OUT_DIR / f"{ts}_{DATASETS_USED}_per_class_metrics.csv"
per_class_df.to_csv(per_class_csv_path, index=False)
print(f"Saved per-class metrics to {per_class_csv_path}")

# and a tidy coefficients CSV — one row per (class, sense) — matches your paper's coefficient tables directly
coef_rows = []
for class_idx, name in enumerate(class_display_names):
    for feat_idx, feature in enumerate(sense_cols_prefixed):
        coef_rows.append({
            "class_name": name,
            "sense": feature.replace(use_what, "").replace(".mean", ""),
            "coef_mean": coef_array[:, class_idx, feat_idx].mean(),
            "coef_std": coef_array[:, class_idx, feat_idx].std(),
        })

coef_df = pd.DataFrame(coef_rows)
coef_csv_path = OUT_DIR / f"{ts}_{DATASETS_USED}_coefficients.csv"
coef_df.to_csv(coef_csv_path, index=False)
print(f"Saved coefficients to {coef_csv_path}")

# %%

# ============================================================
# Coefficient plot — one panel per class, fully dynamic
# ============================================================

# distinct colors regardless of n_classes
palette = sns.color_palette("colorblind", n_classes)

fig, axes = plt.subplots(1, n_classes, figsize=(3.3 * n_classes, 4), sharey=True, dpi=500)
if n_classes == 1:
    axes = [axes]  # keep iterable if somehow only one class

# shared x-axis limits derived from actual coefficient range, with padding,
# instead of a hardcoded (-6.5, 7) that assumed a specific dataset
coef_min, coef_max = coef_array.min(), coef_array.max()
pad = 0.15 * (coef_max - coef_min)
xlim = (coef_min - pad, coef_max + pad)

for ax, class_idx, name, color in zip(axes, range(n_classes), class_display_names, palette):
    means = coef_array[:, class_idx, :].mean(axis=0)
    stds = coef_array[:, class_idx, :].std(axis=0)
    y_display = np.arange(len(senses))[::-1]  # reverses display order only

    # mean markers: de-emphasize if |mean| < 1*std (i.e. consistent with zero)
    reliable = np.abs(means) >= stds

    ax.scatter(
        means[reliable], y_display[reliable],
        color=color, edgecolor='black', linewidth=1,
        s=80, alpha=1, zorder=2,
    )
    ax.scatter(
        means[~reliable], y_display[~reliable],
        color='white', edgecolor='black', linewidth=1,
        s=80, zorder=2,
    )

    ax.errorbar(
        means, y_display, xerr=stds,
        fmt='none', ecolor='black', elinewidth=.5,
        capsize=13, capthick=.5, zorder=1,
    )

    ax.axvline(0, color='0.1', linestyle='--', zorder=0.5)
    ax.set_title(name)
    ax.set_xlim(xlim)

    ax.grid(True, color='0.4', linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

axes[0].set_yticks(range(len(senses)))
axes[0].set_yticklabels([x.title() for x in senses[::-1]], fontsize=12)
fig.supxlabel("Feature coefficient (log-odds)")
plt.tight_layout()
plt.savefig(f"../figs/{ts}_{DATASETS_USED}_coef_plot.png", bbox_inches='tight')
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

##### ROBUSTNESS CHECK: LENGTH / COVERAGE / DIVERSITY CONFOUNDS #####

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

coverage_col = "perc_valid_scores_visual.mean"
lines = []  # collect everything to write at the end

def log(msg=""):
    print(msg)
    lines.append(str(msg))

log("="*70)
log("ROBUSTNESS CHECK: LENGTH, COVERAGE, AND LEXICAL DIVERSITY")
log(f"Generated: {ts}")
log("="*70)

# ------------------------------------------------------------------
# 1. Within-Chicago: does token count predict visual score?
# ------------------------------------------------------------------
log("\n--- 1. Within-Chicago regression: n_tokens -> visual score ---\n")

chic_df = datasets["chicago"]
X_len = sm.add_constant(chic_df["n_tokens"])
y_visual = chic_df["avg_matched_visual.mean"]
model_chic = sm.OLS(y_visual, X_len).fit()

log(f"beta (n_tokens) = {model_chic.params['n_tokens']:.4e}")
log(f"p-value         = {model_chic.pvalues['n_tokens']:.4g}")
log(f"R-squared       = {model_chic.rsquared:.4f}")
log(f"n               = {int(model_chic.nobs)}")

# ------------------------------------------------------------------
# 2. Pooled regression: corpus identity + token count -> visual score
# ------------------------------------------------------------------
log("\n--- 2. Pooled regression: corpus + n_tokens -> visual score ---\n")

all_dfs = []
for name, df in datasets.items():
    d = df[["avg_matched_visual.mean", "n_tokens"]].copy()
    d["corpus"] = name
    all_dfs.append(d)
pooled = pd.concat(all_dfs, ignore_index=True)

model_pooled = smf.ols('Q("avg_matched_visual.mean") ~ n_tokens + C(corpus)', data=pooled).fit()

log(f"n_tokens: beta = {model_pooled.params['n_tokens']:.4e}, p = {model_pooled.pvalues['n_tokens']:.4g}")
log(f"n = {int(model_pooled.nobs)}\n")
log("Corpus effects (relative to reference corpus):")
for term in model_pooled.params.index:
    if term.startswith("C(corpus)"):
        name = term.replace("C(corpus)[T.", "").rstrip("]")
        log(f"  {name:25s} beta = {model_pooled.params[term]:+.4f}   p = {model_pooled.pvalues[term]:.4g}")

# ------------------------------------------------------------------
# 3. Within-corpus correlations: n_tokens vs. visual score
# ------------------------------------------------------------------
log("\n--- 3. Within-corpus Spearman correlations: n_tokens vs. visual score ---\n")

for name, df in datasets.items():
    rho, pval = stats.spearmanr(df["n_tokens"], df["avg_matched_visual.mean"])
    log(f"{name:25s}  rho={rho:+.3f}   p={pval:.4g}   n={len(df)}")

# ------------------------------------------------------------------
# 4. Coverage & lexical diversity: means/SDs, plus correlation with visual score
# ------------------------------------------------------------------
log("\n--- 4. Coverage & MSTTR: descriptive stats and correlation with visual score ---\n")

for name, df in datasets.items():
    log(f"\n{name}")
    log("-" * len(name))

    # descriptive stats
    cov_mean, cov_sd = df[coverage_col].mean(), df[coverage_col].std()
    valid_msttr = df["msttr_lemmas"].notna()
    msttr_mean = df.loc[valid_msttr, "msttr_lemmas"].mean()
    msttr_sd = df.loc[valid_msttr, "msttr_lemmas"].std()

    log(f"  Coverage (perc_valid_scores_visual):  mean={cov_mean:.3f}  sd={cov_sd:.3f}  n={len(df)}")
    log(f"  MSTTR (lexical diversity):             mean={msttr_mean:.3f}  sd={msttr_sd:.3f}  n={valid_msttr.sum()} (of {len(df)})")

    # correlations with visual score
    rho_cov, p_cov = stats.spearmanr(df[coverage_col], df["avg_matched_visual.mean"])
    rho_msttr, p_msttr = stats.spearmanr(
        df.loc[valid_msttr, "msttr_lemmas"], df.loc[valid_msttr, "avg_matched_visual.mean"]
    )
    log(f"  Coverage vs. visual score:  rho={rho_cov:+.3f}  p={p_cov:.4g}")
    log(f"  MSTTR vs. visual score:     rho={rho_msttr:+.3f}  p={p_msttr:.4g}")

# ------------------------------------------------------------------
# Save everything to a single txt file
# ------------------------------------------------------------------
log("\n" + "="*70)
log("END OF ROBUSTNESS CHECK")

output_path = OUT_DIR / f"{ts}_robustness_check.txt"
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"\nSaved full robustness check to {output_path}")
# %%
