
# %%
import json
import time
import numpy as np
from pathlib import Path
import openpyxl
import re
from math import pi

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

ts = time.strftime("%Y-%m-%d")
print(f"Timestamp: {ts}")

CWD = Path(__file__).parent
DATA_PATH = CWD.parent / "data" / "lemmatized_data" / "scored_data"
FIGS = CWD.parent / "figs"
OUT_DIR = CWD / "OUT_DIR"
CLEAN_DIR = OUT_DIR / "clean_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# %%

datasets = {}

for path in sorted(Path(DATA_PATH).glob("*.json")):
    name = path.stem.replace("_with_scores", "").replace("_lemmatized", "")
    datasets[name] = pd.read_json(path, orient='records', lines=True)
    print(f"Loaded {name} dataset with {len(datasets[name])} entries.")

datasets.keys()

# %%
### Build author labels ###

datasets["simplestories"]["author"] = (datasets["simplestories"]["persona"] + "_" + datasets["simplestories"]["style"])

storyscope = datasets.pop("storyscope")
storyscope["model"] = storyscope["work_id"].str.extract(r"_([a-zA-Z]+)$")
storyscope["author"] = storyscope["model"] + "_" + storyscope["human_author"]

print(storyscope["model"].value_counts(dropna=False))

# %%
### Split storyscope into per-model datasets ###

for model, group in storyscope.groupby("model"):
    datasets[f"storyscope_{model}"] = group.drop(columns="model").reset_index(drop=True)

storyscope_keys = [k for k in datasets if k.startswith("storyscope_")]
print(storyscope_keys)

# sanity check
for key in ["simplestories", *storyscope_keys]:
    vc = datasets[key]["author"].value_counts()
    print(f"\n{key} — authors with >10 stories:")
    print(vc[vc > 10])

# %%

# Add chicago metadata

# meta_chic
meta = pd.read_excel(CWD.parent / "data" / "CHICAGO_MEASURES_MARCH24.xlsx")
meta = meta[["BOOK_ID", "AUTH_FIRST", "AUTH_LAST", "WORDCOUNT", "PUBL_DATE", "LIBRARIES", "RATING_COUNT", ]]
meta.columns = ["work_id", "author_first", "author_last", "text_length", "year", "libraries", "rating_count"]
meta["author"] = meta["author_first"] + " " + meta["author_last"]
meta.drop(columns=["author_first", "author_last"], inplace=True)
chic = datasets["chicago"].merge(meta, how='left', on='work_id')
# rename cols
chic.rename(columns={col: "avg_matched_" + col.replace("_mean", ".mean") for col in chic.columns if col.endswith('_mean') and any(sense in col for sense in ['auditory', 'gustatory', 'haptic', 'interoceptive', 'olfactory', 'visual'])}, inplace=True)

datasets["chicago"] = chic
datasets.keys()

# %%

##### sense columns and derived metrics #####

# decide which columns to use, here senses + normalized & engagement metrics
sense_cols = ["auditory.mean", "gustatory.mean", "olfactory.mean", "haptic.mean", "visual.mean", "interoceptive.mean"]
USE_WHAT = "avg_matched_" # set this to total, avg_matched, or normalized
sense_cols_prefixed = [USE_WHAT + col for col in sense_cols]


# for each dataset, we want to add entropy
def add_entropy(df, sense_cols_prefixed):
    df[f'{USE_WHAT}sense_sum'] = df[sense_cols_prefixed].sum(axis=1)
    # add a column for the percent of each sense
    for sense in sense_cols_prefixed:
        colname = sense.replace('.mean', '_percent')
        df[f"{colname}"] = df[sense] / df[f'{USE_WHAT}sense_sum']
    # entropy of the sense distribution (we use the percent columns for this, since they sum to 1)
    df['sense_entropy'] = df[[col for col in df.columns if col.endswith('_percent')]].apply(lambda x: entropy(x, base=2), axis=1)
    return df

# apply
for name, df in datasets.items():
    datasets[name] = add_entropy(df, sense_cols_prefixed)

# define additional sense columns to use
add_sense_cols = [f'{USE_WHAT}sense_sum', 'sense_entropy'] 
percent_sense_cols = [col for col in df.columns if col.endswith('_percent')] # percent show some of the same info so we skip them for now


# %%

### REMOVE 0 IF ACROSS ALL FEATS ####

# check before filtering
# drop rows with zero total sense score (degenerate/near-empty texts)
for ds in list(datasets.keys()):
    zero_mask = datasets[ds][sense_cols_prefixed].sum(axis=1) == 0
    print(f"N zeroes {ds}: {zero_mask.sum()} out of {len(zero_mask)}")
    datasets[ds] = datasets[ds][~zero_mask].reset_index(drop=True)
    print(f"After filtering: {len(datasets[ds])} (dropped {zero_mask.sum()})")


# %%
### Summary statistics per dataset ###

output_path = OUT_DIR / f"{ts}_sense_score_summary.txt"
lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))

rows = []

for name, df in datasets.items():
    log(f"Dataset: {name}")
    log("-" * (len(name) + 9))
    log(f"  N texts: {len(df)}")

    coverage_col = "perc_valid_scores_" + sense_cols[0].replace(USE_WHAT, "")
    cov_mean, cov_sd = df[coverage_col].mean(), df[coverage_col].std()
    log(f"  coverage        mean={cov_mean:.3f}  sd={cov_sd:.3f}")

    for col, label in [("n_tokens", "n_tokens"),
                        ("msttr_lemmas", "msttr_lemmas"),
                        ("unique_vs_all_lemmas", "unique_vs_all")]:
        if col in df.columns:
            log(f"  {label:15s} mean={df[col].mean():.3f}  sd={df[col].std():.3f}")

    for sense in sense_cols_prefixed:
        label = sense.replace(USE_WHAT, "").replace(".mean", "")
        rho, pval = spearmanr(df[sense], df["sense_entropy"])
        log(f"  {label:15s}  mean={df[sense].mean():.3f}  sd={df[sense].std():.3f}   "
            f"entropy corr: rho={rho:+.3f} p={pval:.3g}")
        rows.append({
            "dataset": name, "sense": label,
            "score_mean": df[sense].mean(), "score_sd": df[sense].std(),
            "coverage_mean": cov_mean, "coverage_sd": cov_sd,
            "entropy_corr_rho": rho, "entropy_corr_p": pval,
        })

    log(f"  {'sense_entropy':15s}  mean={df['sense_entropy'].mean():.3f}  sd={df['sense_entropy'].std():.3f}")
    log("=" * 60)

output_path.write_text("\n".join(lines))
print(f"\nSaved summary to {output_path}")

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_DIR / f"{ts}_sense_score_summary.csv", index=False)
print(f"Saved tidy summary to {OUT_DIR / f'{ts}_sense_score_summary.csv'}")

# %%

# we want to do a kde plot of the distribution of each sense score for fanfiction and chicago, to see how they compare
senses = ["auditory", "gustatory", "olfactory", "haptic", "visual", "interoceptive"]

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
axes = axes.flatten()

emphasize = {"chicago", "fanfics", "simplestories"}

main_styles = {"chicago":       {"color": "0.05", "linestyle": "-"},
            "fanfics":       {"color": "0.3", "linestyle": "--"},
            "simplestories": {"color": "0.45", "linestyle": ":"}}

storyscope_names = [name for name in datasets if name.startswith("storyscope_")]
storyscope_palette = sns.color_palette("husl", len(storyscope_names))
storyscope_colors = dict(zip(storyscope_names, storyscope_palette))

# keep track of handles/labels separately for main vs. storyscope, in plotting order
main_handles, main_labels = [], []
storyscope_handles, storyscope_labels = [], []

for i, sense in enumerate(senses):
    ax = axes[i]
    col = f'{USE_WHAT}{sense}.mean'

    pooled_vals = pd.concat([df[col] for df in datasets.values()])
    lo, hi = pooled_vals.quantile([0.002, 0.998])  # loosened from 0.005/0.995

    for name, df in datasets.items():
        if name in emphasize:
            style = main_styles[name]
            line = sns.kdeplot(
                data=df, x=col, label=name.capitalize(),
                fill=True, alpha=0.15,
                color=style["color"], linestyle=style["linestyle"],
                linewidth=2.2,
                ax=ax, clip=(lo, hi))
        else:
            line = sns.kdeplot(
                data=df, x=col, label=name.capitalize(),
                fill=False, alpha=0.85,
                color=storyscope_colors[name],
                linewidth=1.2,
                ax=ax, clip=(lo, hi))

    ax.set_xlim(lo, hi)
    ax.set_xlabel(sense.capitalize())
    ax.set_ylabel('Density' if i % 3 == 0 else '')

    # grab handles/labels only once, from the first subplot
    if i == 0:
        all_handles, all_labels = ax.get_legend_handles_labels()
        for h, l in zip(all_handles, all_labels):
            if l.lower() in emphasize:
                main_handles.append(h)
                main_labels.append(l)
            else:
                storyscope_handles.append(h)
                storyscope_labels.append(l)

for ax in axes:
    if ax.get_legend(): ax.get_legend().remove()

# two explicit legend rows: main categories on top, Storyscope models below
leg1 = fig.legend(main_handles, main_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.12), ncol=len(main_handles), frameon=False)
fig.add_artist(leg1)
fig.legend(storyscope_handles, [x.replace("_", " ").title() for x in storyscope_labels], loc='upper center',
           bbox_to_anchor=(0.5, 1.06), ncol=len(storyscope_handles), frameon=False)

sns.set_style("whitegrid")
plt.tight_layout()
plt.savefig(FIGS / f"{ts}_sense_score_distributions_based_on_{USE_WHAT}.png", bbox_inches='tight')
plt.show()

# %%

### CLASSIFICATION ###

# ============================================================
# CONFIG
# ============================================================

# Pick one grouping mode, or loop over all three further down.
#   "four_class"   -> chicago / fanfics / simplestories / storyscope (all models combined)
#   "eight_class"         -> chicago / fanfics / simplestories / storyscope_gpt / storyscope_claude / ...
#   "three_class" -> chicago / fanfics / generated (simplestories + all storyscope models pooled)
MODE = "three_class"

random_state = 42

# %%

# ============================================================
# Build class groups
# ============================================================

def build_class_groups(mode, datasets):
    storyscope_keys = [k for k in datasets if k.startswith("storyscope_")]

    if mode == "four_class":
        storyscope_combined = pd.concat([datasets[k] for k in storyscope_keys], axis=0)
        return {"chicago": datasets["chicago"],
            "fanfics": datasets["fanfics"],
            "simplestories": datasets["simplestories"],
            "storyscope": storyscope_combined}

    elif mode == "eight_class":
        return {k: df for k, df in datasets.items() if k != "storyscope"}

    elif mode == "three_class":
        # then we combine all the generated text as one
        generated_combined = pd.concat([datasets["simplestories"], *[datasets[k] for k in storyscope_keys]], axis=0)
        return {"chicago": datasets["chicago"],
            "fanfics": datasets["fanfics"],
            "generated": generated_combined}

    else:
        raise ValueError(f"Unknown mode: {mode}")


class_groups = build_class_groups(MODE, datasets)
class_names = sorted(class_groups.keys())
label_map = {name: i for i, name in enumerate(class_names)}
print(f"Mode: {MODE}")
print("Label mapping:", label_map)

# %%
# ============================================================
# Assemble balanced dataset
# ============================================================

def build_balanced_dataset(class_groups, label_map, sense_cols, random_state, out_dir, ts, mode):
    cols = sense_cols + ["label", "author", "n_tokens", "sense_entropy"]
    target_n = min(len(df) for df in class_groups.values())
    smallest = min(class_groups, key=lambda k: len(class_groups[k]))
    print(f"Downsampling every class to n={target_n} (smallest: {smallest})")

    pieces = []
    for name, df in class_groups.items():
        d = df.copy()
        d["label"] = label_map[name]
        n = min(target_n, len(d))
        if n < target_n:
            print(f"WARNING: {name} has only {len(d)} rows, less than target {target_n}")
        pieces.append(d.sample(n=n, random_state=random_state)[cols])

    together = pd.concat(pieces, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # dummy-ids for any still-missing authors (belt-and-braces; should mostly be handled upstream now)
    missing_mask = together["author"].isna()
    if missing_mask.any():
        together.loc[missing_mask, "author"] = [f"missing_author_{i}" for i in range(missing_mask.sum())]
        print(f"Filled {missing_mask.sum()} missing authors with dummy ids")

    # log summary
    summary_lines = [
        f"Balanced dataset stats:\n{together['label'].value_counts().sort_index()}",
        f"Number of unique authors: {together['author'].nunique()}",
        f"N unique authors per label: {together.groupby('label')['author'].nunique()}",
        f"Avg number of samples per author: {together.groupby('author').size().mean():.2f}",
        f"Avg textlength per label: {together.groupby('label')['n_tokens'].mean()}",
        "==========",
    ]
    for line in summary_lines:
        print(line)
    (out_dir / f"{ts}_{mode}_classification_data_summary.txt").write_text("\n".join(summary_lines))

    together = together.drop(columns=["n_tokens"])
    together["author"] = together["author"].astype("object")

    out_path = out_dir / f"{mode}_classification_dataset.csv.gz"
    together.to_csv(out_path, index=False, compression="gzip")
    print(f"Saved balanced dataset to {out_path}")

    return together


together = build_balanced_dataset(class_groups, label_map, sense_cols_prefixed, random_state, OUT_DIR, ts, MODE)

# %%
# ============================================================
# Run Cross-validated classification
# ============================================================

def run_classification(together, sense_cols, label_map, random_state=42, n_splits=5):
    X = together[sense_cols]
    y = together["label"]
    groups = together["author"]

    class_labels = sorted(y.unique())
    n_classes = len(class_labels)
    inv_label_map = {v: k for k, v in label_map.items()}
    class_display_names = [inv_label_map[c].replace("_", " ").title() for c in class_labels]
    print(f"Classes found: {dict(zip(class_labels, class_display_names))}")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=1000, random_state=random_state),
    )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = {k: [] for k in ["fold", "accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]}
    per_class_scores = {f"{c}_{m}": [] for c in class_labels for m in ["precision", "recall", "f1"]}
    all_coefs, all_y_true, all_y_pred = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        roc_auc = roc_auc_score(label_binarize(y_test, classes=class_labels), y_proba, multi_class="ovr")
        report = classification_report(y_test, y_pred, output_dict=True)

        for c in class_labels:
            key = str(c)
            per_class_scores[f"{c}_precision"].append(report[key]["precision"])
            per_class_scores[f"{c}_recall"].append(report[key]["recall"])
            per_class_scores[f"{c}_f1"].append(report[key]["f1-score"])

        scores["fold"].append(fold)
        scores["accuracy"].append(model.score(X_test, y_test))
        scores["macro_precision"].append(report["macro avg"]["precision"])
        scores["macro_recall"].append(report["macro avg"]["recall"])
        scores["macro_f1"].append(report["macro avg"]["f1-score"])
        scores["roc_auc_ovr"].append(roc_auc)
        all_coefs.append(model.named_steps["logisticregression"].coef_)

    coef_array = np.array(all_coefs)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=class_labels)

    print("\n--- Average Performance Across Folds ---")
    for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]:
        print(f"{metric}: {np.mean(scores[metric]):.4f} ± {np.std(scores[metric]):.4f}")

    print("\n--- Per-Class Performance (mean ± std) ---")
    for c, name in zip(class_labels, class_display_names):
        print(f"\nClass {c} ({name}):")
        for metric in ["precision", "recall", "f1"]:
            key = f"{c}_{metric}"
            print(f"  {metric}: {np.mean(per_class_scores[key]):.4f} ± {np.std(per_class_scores[key]):.4f}")

    return {
        "class_labels": class_labels,
        "class_display_names": class_display_names,
        "scores": scores,
        "per_class_scores": per_class_scores,
        "coef_array": coef_array,
        "cm": cm,
    }


results_raw = run_classification(together, sense_cols_prefixed, label_map, random_state=random_state)


# %%

# ============================================================
# Structured output: per-fold metrics, per-class metrics, coefficients
# ============================================================


def save_classification_outputs(results_raw, label_map, sense_cols, mode, ts, out_dir, figs_dir, use_what):
    class_labels = results_raw["class_labels"]
    class_display_names = results_raw["class_display_names"]
    scores = results_raw["scores"]
    per_class_scores = results_raw["per_class_scores"]
    coef_array = results_raw["coef_array"]
    cm = results_raw["cm"]
    n_classes = len(class_labels)

    # coefficients per class
    for class_idx in range(coef_array.shape[1]):
        name = class_display_names[class_idx]
        print(f"\nClass {class_labels[class_idx]} ({name}) coefficients (mean ± std):")
        print(pd.DataFrame({
            "mean": coef_array[:, class_idx, :].mean(axis=0),
            "std": coef_array[:, class_idx, :].std(axis=0),
        }, index=sense_cols).sort_values("mean", ascending=False))

    # confusion matrix plot
    fig_size = max(3, n_classes * 0.9)
    plt.figure(figsize=(fig_size, fig_size), dpi=500)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_display_names, yticklabels=class_display_names)
    plt.xlabel(r"$\bf{Predicted}$")
    plt.ylabel(r"$\bf{True}$")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figs_dir / f"{ts}_{mode}_confusion_matrix.png")
    plt.show()

    # structured JSON
    results = {
        "timestamp": ts,
        "dataset_config": mode,
        "n_classes": n_classes,
        "class_labels": [int(c) for c in class_labels],
        "class_display_names": class_display_names,
        "label_map": label_map,
        "n_folds": len(scores["fold"]),
        "overall_metrics": {
            m: {"mean": float(np.mean(scores[m])), "std": float(np.std(scores[m]))}
            for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]
        },
        "per_fold_metrics": {
            m: [float(v) for v in scores[m]]
            for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]
        },
        "per_class_metrics": {
            name: {
                m: {
                    "mean": float(np.mean(per_class_scores[f"{c}_{m}"])),
                    "std": float(np.std(per_class_scores[f"{c}_{m}"])),
                }
                for m in ["precision", "recall", "f1"]
            }
            for c, name in zip(class_labels, class_display_names)
        },
        "coefficients": {
            name: {
                feat: {
                    "mean": float(coef_array[:, idx, fi].mean()),
                    "std": float(coef_array[:, idx, fi].std()),
                }
                for fi, feat in enumerate(sense_cols)
            }
            for idx, name in enumerate(class_display_names)
        },
        "confusion_matrix": cm.tolist(),
    }
    results_path = out_dir / f"{ts}_{mode}_classification_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved structured results to {results_path}")

    # tidy per-class CSV
    per_class_df = pd.DataFrame([
        {
            "class_label": c, "class_name": name, "metric": m,
            "mean": np.mean(per_class_scores[f"{c}_{m}"]),
            "std": np.std(per_class_scores[f"{c}_{m}"]),
        }
        for c, name in zip(class_labels, class_display_names)
        for m in ["precision", "recall", "f1"]
    ])
    per_class_df.to_csv(out_dir / f"{ts}_{mode}_per_class_metrics.csv", index=False)

    # tidy coefficients CSV
    coef_df = pd.DataFrame([
        {
            "class_name": name,
            "sense": feat.replace(use_what, "").replace(".mean", ""),
            "coef_mean": coef_array[:, idx, fi].mean(),
            "coef_std": coef_array[:, idx, fi].std(),
        }
        for idx, name in enumerate(class_display_names)
        for fi, feat in enumerate(sense_cols)
    ])
    coef_df.to_csv(out_dir / f"{ts}_{mode}_coefficients.csv", index=False)
    print(f"Saved per-class metrics and coefficients CSVs for mode={mode}")


save_classification_outputs(results_raw, label_map, sense_cols_prefixed, MODE, ts, OUT_DIR, FIGS, USE_WHAT)


# %%

# ============================================================
# Coefficient plot — one panel per class, fully dynamic
# ============================================================

# %%
def plot_coefficients(results_raw, sense_cols, use_what, mode, ts, figs_dir):
    coef_array = results_raw["coef_array"]
    class_display_names = results_raw["class_display_names"]
    n_classes = len(class_display_names)

    # human-readable sense names, derived from the actual columns used
    senses = [c.replace(use_what, "").replace(".mean", "") for c in sense_cols]

    palette = sns.color_palette("colorblind", n_classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(3.3 * n_classes, 4), sharey=True, dpi=500)
    if n_classes == 1:
        axes = [axes]

    coef_min, coef_max = coef_array.min(), coef_array.max()
    pad = 0.15 * (coef_max - coef_min)
    xlim = (coef_min - pad, coef_max + pad)

    for ax, class_idx, name, color in zip(axes, range(n_classes), class_display_names, palette):
        means = coef_array[:, class_idx, :].mean(axis=0)
        stds = coef_array[:, class_idx, :].std(axis=0)
        y_display = np.arange(len(senses))[::-1]

        reliable = np.abs(means) >= stds

        ax.scatter(means[reliable], y_display[reliable],
                   color=color, edgecolor="black", linewidth=1, s=80, alpha=1, zorder=2)
        ax.scatter(means[~reliable], y_display[~reliable],
                   color="white", edgecolor="black", linewidth=1, s=80, zorder=2)
        ax.errorbar(means, y_display, xerr=stds,
                    fmt="none", ecolor="black", elinewidth=.5, capsize=13, capthick=.5, zorder=1)

        ax.axvline(0, color="0.1", linestyle="--", zorder=0.5)
        ax.set_title(name)
        ax.set_xlim(xlim)
        ax.grid(True, color="0.4", linewidth=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_yticks(range(len(senses)))
    axes[0].set_yticklabels([s.title() for s in senses[::-1]], fontsize=12)
    fig.supxlabel("Feature coefficient (log-odds)")
    plt.tight_layout()

    out_path = figs_dir / f"{ts}_{mode}_coef_plot.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved coefficient plot to {out_path}")


plot_coefficients(results_raw, sense_cols, USE_WHAT, MODE, ts, FIGS)




# %%

##### ROBUSTNESS CHECK: LENGTH / COVERAGE / DIVERSITY CONFOUNDS #####

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
