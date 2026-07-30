
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
from scipy.stats import entropy, spearmanr, levene

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

ts = time.strftime("%Y-%m-%d")
print(f"Timestamp: {ts}")

CWD = Path(__file__).parent
#DATA_PATH = CWD.parent / "data" / "lemmatized_data" / "scored_data"
DATA_PATH = CWD.parent / "data" / "checkpoint"
FIGS = CWD.parent / "figs"
OUT_DIR = CWD / "OUT_DIR"
OUT_DIR.mkdir(parents=True, exist_ok=True)



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
### SUMMARY STATISTICS / DATASET ###
def summarize_groups(groups_dict, sense_cols_prefixed, sense_cols, use_what):
    """
    groups_dict: dict of name -> dataframe (e.g. datasets, or fandom groups within fanfics)
    Returns lines, the printable/log text ready to save to a txt file.
    """
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    for name, df in groups_dict.items():
        log(f"Group: {name}")
        log("-" * (len(name) + 7))
        log(f"  N texts: {len(df)}")
        log(f"  N unique authors: {df['author'].nunique()}")

        coverage_col = "perc_valid_scores_" + sense_cols[0].replace(use_what, "")
        cov_mean, cov_sd = df[coverage_col].mean(), df[coverage_col].std()
        log(f"  coverage        mean={cov_mean:.3f}  sd={cov_sd:.3f}")

        for col, label in [("n_tokens", "n_tokens"),
                            ("msttr_lemmas", "msttr_lemmas"),
                            ("unique_vs_all_lemmas", "unique_vs_all")]:
            if col in df.columns:
                log(f"  {label:15s} mean={df[col].mean():.3f}  sd={df[col].std():.3f}")

        for sense in sense_cols_prefixed:
            label = sense.replace(use_what, "").replace(".mean", "")
            rho, pval = spearmanr(df[sense], df["sense_entropy"])
            log(f"  {label:15s}  mean={df[sense].mean():.3f}  sd={df[sense].std():.3f}   "
                f"entropy corr: rho={rho:+.3f} p={pval:.3g}")

        log(f"  {'sense_entropy':15s}  mean={df['sense_entropy'].mean():.3f}  sd={df['sense_entropy'].std():.3f}")
        log("=" * 60)

    return lines


def save_summary(lines, out_dir, ts, tag):
    output_path = out_dir / f"{ts}_{tag}_summary.txt"
    output_path.write_text("\n".join(lines))
    print(f"\nSaved summary to {output_path}")


# per-dataset summary (unchanged behavior)
lines = summarize_groups(datasets, sense_cols_prefixed, sense_cols, USE_WHAT)
save_summary(lines, OUT_DIR, ts, tag="sense_score")

# per-fandom summary (new)
fandom_groups = {fandom: group for fandom, group in datasets["fanfics"].groupby("fandom_label")}
fandom_lines = summarize_groups(fandom_groups, sense_cols_prefixed, sense_cols, USE_WHAT)
save_summary(fandom_lines, OUT_DIR, ts, tag="fandom_sense_score")

# %%

### PLOT KDE ####

# we want to do a kde plot of the distribution of each sense score for fanfiction and chicago, to see how they compare
senses = ["auditory", "gustatory", "olfactory", "haptic", "visual", "interoceptive"]

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
axes = axes.flatten()

emphasize = {"chicago", "fanfics", "simplestories"}

main_styles = {"chicago":       {"color": "0.05", "linestyle": "-"},
            "fanfics":       {"color": "0.3", "linestyle": "--"},
            "simplestories": {"color": "0.45", "linestyle": ":"}}

# exclude the combined "storyscope" entry — per-model versions are shown individually,
# and including both would double-count rows in the pooled quantile calc below
plot_datasets = {k: v for k, v in datasets.items() if k != "storyscope"}

storyscope_names = [name for name in plot_datasets if name.startswith("storyscope_")]
storyscope_palette = sns.color_palette("husl", len(storyscope_names))
storyscope_colors = dict(zip(storyscope_names, storyscope_palette))

# keep track of handles/labels separately for main vs. storyscope, in plotting order
main_handles, main_labels = [], []
storyscope_handles, storyscope_labels = [], []

for i, sense in enumerate(senses):
    ax = axes[i]
    col = f'{USE_WHAT}{sense}.mean'

    pooled_vals = pd.concat([df[col] for df in plot_datasets.values()])
    lo, hi = pooled_vals.quantile([0.002, 0.998])  # loosened from 0.005/0.995

    for name, df in plot_datasets.items():
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
# Build class groups
# ============================================================

def build_class_groups(mode, datasets):
    storyscope_keys = [k for k in datasets if k.startswith("storyscope_")]

    if mode == "four_class":
        storyscope_combined = pd.concat([datasets[k].assign(model=k) for k in storyscope_keys], axis=0, ignore_index=True)
        return {"chicago": datasets["chicago"],
            "fanfics": datasets["fanfics"],
            "simplestories": datasets["simplestories"],
            "storyscope": storyscope_combined}

    elif mode == "three_class":
        # tag each generated source before pooling, so we can stratify sampling across them
        generated_pieces = []
        simplestories_tagged = datasets["simplestories"].copy()
        simplestories_tagged["source"] = "simplestories"
        generated_pieces.append(simplestories_tagged)

        for k in storyscope_keys:
            piece = datasets[k].copy()
            piece["source"] = k  # e.g. "storyscope_claude"
            generated_pieces.append(piece)

        generated_combined = pd.concat(generated_pieces, axis=0, ignore_index=True)
        return {"chicago": datasets["chicago"],
            "fanfics": datasets["fanfics"],
            "generated": generated_combined}

    else:
        raise ValueError(f"Unknown mode: {mode}")


# %%
# ============================================================
# Assemble balanced dataset
# ============================================================

# ============================================================
# Assemble balanced dataset (now with optional stratified sampling)
# ============================================================

def stratified_sample(df, n, stratify_col, random_state):
    """
    Sample n rows from df, split as evenly as possible across the levels of
    stratify_col (e.g. fandom, or model). Falls back to plain random sampling
    if stratify_col is None or not present in df.
    """
    if stratify_col is None or stratify_col not in df.columns:
        return df.sample(n=n, random_state=random_state)

    levels = df[stratify_col].dropna().unique()
    n_levels = len(levels)
    if n_levels == 0:
        return df.sample(n=n, random_state=random_state)

    base_n = n // n_levels
    remainder = n % n_levels  # distribute leftover rows across the first few levels

    pieces = []
    for i, level in enumerate(sorted(levels)):
        sub = df[df[stratify_col] == level]
        target = base_n + (1 if i < remainder else 0)
        take = min(target, len(sub))
        if take < target:
            print(f"    WARNING: level '{level}' of '{stratify_col}' has only {len(sub)} rows, "
                  f"wanted {target}")
        pieces.append(sub.sample(n=take, random_state=random_state))

    result = pd.concat(pieces, axis=0)
    # if any level came up short, top up from the remaining pool (excluding what's already taken)
    shortfall = n - len(result)
    if shortfall > 0:
        remaining_pool = df.drop(result.index)
        top_up = remaining_pool.sample(n=min(shortfall, len(remaining_pool)), random_state=random_state)
        result = pd.concat([result, top_up], axis=0)

    return result


def build_balanced_dataset(class_groups, label_map, sense_cols, random_state, out_dir, ts, mode,
                            extra_cols=None, stratify_cols=None):
    """
    stratify_cols: optional dict mapping class name -> column name to stratify by
                   when sampling that class (e.g. {"fanfics": "fandom_label",
                   "storyscope": "model"}). Classes not in this dict are sampled
                   with plain random sampling.
    """
    stratify_cols = stratify_cols or {}
    cols = sense_cols + ["label", "author", "n_tokens", "sense_entropy"] + (extra_cols or [])
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

        stratify_col = stratify_cols.get(name)
        if stratify_col:
            print(f"  Sampling '{name}' stratified by '{stratify_col}'")
            sampled = stratified_sample(d, n, stratify_col, random_state)
        else:
            sampled = d.sample(n=n, random_state=random_state)

        pieces.append(sampled[cols])

    together = pd.concat(pieces, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    missing_mask = together["author"].isna()
    if missing_mask.any():
        together.loc[missing_mask, "author"] = [f"missing_author_{i}" for i in range(missing_mask.sum())]
        print(f"Filled {missing_mask.sum()} missing authors with dummy ids")

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

    together["author"] = together["author"].astype("object")

    out_path = out_dir / f"{mode}_classification_dataset.csv.gz"
    save_cols = [c for c in together.columns if c != "lemmatized_text"]
    together[save_cols].drop(columns=["n_tokens"]).to_csv(out_path, index=False, compression="gzip")
    print(f"Saved balanced dataset to {out_path}")

    return together

# %%
# ============================================================
# Cross-validated classification
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
        "cm": cm}

# %%

# ============================================================
# Structured output: per-fold metrics, per-class metrics, coefficients
# ============================================================


def save_classification_outputs(results_raw, label_map, sense_cols, mode, ts, out_dir, figs_dir, use_what, print_top_n=None):
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
        n_print = print_top_n if print_top_n is not None else len(sense_cols)
        print(f"\nClass {class_labels[class_idx]} ({name}) coefficients (mean ± std):")
        print(pd.DataFrame({
            "mean": coef_array[:, class_idx, :].mean(axis=0),
            "std": coef_array[:, class_idx, :].std(axis=0),
        }, index=sense_cols).sort_values("mean", ascending=False).head(n_print))

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


# %%
# ============================================================
# Build Pipeline
# ============================================================

# ============================================================
# CONFIG
# ============================================================

random_state = 42

# Pick one grouping mode, or loop over all three further down.
#   "four_class"   -> chicago / fanfics / simplestories / storyscope (all models combined)
#   "eight_class"         -> chicago / fanfics / simplestories / storyscope_gpt / storyscope_claude / ...
#   "three_class" -> chicago / fanfics / generated (simplestories + all storyscope models pooled)
MODE = "four_class"

class_groups = build_class_groups(MODE, datasets)
class_names = sorted(class_groups.keys())
label_map = {name: i for i, name in enumerate(class_names)}
print(f"Mode: {MODE}")
print("Label mapping:", label_map)

together = build_balanced_dataset(
    class_groups, label_map, sense_cols_prefixed, random_state, OUT_DIR, ts, MODE,
    extra_cols=["lemmatized_text"],
    stratify_cols={"fanfics": "fandom_label", "generated": "source", "storyscope": "model"})
    
results_raw = run_classification(together, sense_cols_prefixed, label_map, random_state=random_state)
save_classification_outputs(results_raw, label_map, sense_cols_prefixed, MODE, ts, OUT_DIR, FIGS, USE_WHAT)


# %%
# ============================================================
# Build MFW features x 2 and run pipeline
# ============================================================

def build_mfw_features(together, text_col="lemmatized_text", n_features=100, relative=True):
    """
    Builds top-N most-frequent-word features from a column of tokenized/lemmatized text
    (list of tokens per row). Vocabulary is selected on the whole corpus, unsupervised
    (no label information used), consistent with standard stylometric MFW practice.
    """
    vectorizer = CountVectorizer(max_features=n_features, analyzer=lambda tokens: tokens, lowercase=False)
    X_counts = vectorizer.fit_transform(together[text_col])
    feature_names = [f"mfw_{w}" for w in vectorizer.get_feature_names_out()]

    X_df = pd.DataFrame(X_counts.toarray(), columns=feature_names, index=together.index)

    if relative:
        # normalize by text length so longer texts don't just have larger raw counts
        text_lengths = together[text_col].apply(len).replace(0, np.nan)
        X_df = X_df.div(text_lengths, axis=0).fillna(0)

    print(f"Built {len(feature_names)} MFW features (relative={relative})")
    print(f"Top 10 words: {[f.replace('mfw_', '') for f in feature_names[:10]]}")

    return X_df, feature_names


mfw_df, mfw_cols = build_mfw_features(together, n_features=100, relative=True)
together_mfw = pd.concat([together.drop(columns=["lemmatized_text"]), mfw_df], axis=1)


results_raw_mfw = run_classification(together_mfw, mfw_cols, label_map, random_state=random_state)
save_classification_outputs(
    results_raw_mfw, label_map, mfw_cols, mode=f"{MODE}_mfw100",
    ts=ts, out_dir=OUT_DIR, figs_dir=FIGS, use_what="mfw_")


mfw6_df, mfw6_cols = build_mfw_features(together, n_features=6, relative=True)
together_mfw6 = pd.concat([together.drop(columns=["lemmatized_text"]), mfw6_df], axis=1)

results_raw_mfw6 = run_classification(together_mfw6, mfw6_cols, label_map, random_state=random_state)
save_classification_outputs(
    results_raw_mfw6, label_map, mfw6_cols, mode=f"{MODE}_mfw6",
    ts=ts, out_dir=OUT_DIR, figs_dir=FIGS, use_what="mfw_")

# %%

# ============================================================
# Coefficient plot — one panel per class
# ============================================================


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
                    fmt="none", ecolor="black", elinewidth=.5, capsize=15, capthick=.5, zorder=1)

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

# ===========================================================
# TESTING HOW WELL ENTROPY ALONE DOES
# ===========================================================

X_entropy = together[["sense_entropy"]]
y = together["label"]
groups = together["author"]

class_labels = sorted(y.unique())
inv_label_map = {v: k for k, v in label_map.items()}
class_display_names = [inv_label_map[c].replace("_", " ").title() for c in class_labels]

model_entropy = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42))

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

per_class_scores = {f'{c}_{metric}': [] for c in class_labels for metric in ['precision', 'recall', 'f1']}
scores_entropy = []
roc_auc_scores = []
all_y_true, all_y_pred = [], []

for train_idx, test_idx in cv.split(X_entropy, y, groups):
    X_train, X_test = X_entropy.iloc[train_idx], X_entropy.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model_entropy.fit(X_train, y_train)
    y_pred = model_entropy.predict(X_test)
    y_proba = model_entropy.predict_proba(X_test)

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    roc_auc = roc_auc_score(label_binarize(y_test, classes=class_labels), y_proba, multi_class='ovr')
    roc_auc_scores.append(roc_auc)

    report = classification_report(y_test, y_pred, output_dict=True)
    for c in class_labels:
        key = str(c)
        per_class_scores[f'{c}_precision'].append(report[key]['precision'])
        per_class_scores[f'{c}_recall'].append(report[key]['recall'])
        per_class_scores[f'{c}_f1'].append(report[key]['f1-score'])

    scores_entropy.append(model_entropy.score(X_test, y_test))

print(f"Accuracy using entropy alone: {np.mean(scores_entropy):.4f} ± {np.std(scores_entropy):.4f}")
print(f"ROC AUC (OvR): {np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}\n")

print("--- Per-Class Performance (mean ± std) ---")
for c, name in zip(class_labels, class_display_names):
    print(f"\nClass {c} ({name}):")
    for metric in ['precision', 'recall', 'f1']:
        key = f'{c}_{metric}'
        print(f"  {metric}: {np.mean(per_class_scores[key]):.4f} ± {np.std(per_class_scores[key]):.4f}")

cm = confusion_matrix(all_y_true, all_y_pred, labels=class_labels)
print("\nConfusion matrix:")
print(cm)





# %%
# ===========================================================
# SHARED LOGGING HELPER
# ===========================================================

def make_logger():
    """Returns (log, get_lines) — log() prints and buffers; get_lines() returns the buffer."""
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(str(msg))
    return log, lines


def save_log(lines, out_dir, ts, tag):
    output_path = out_dir / f"{ts}_{tag}.txt"
    output_path.write_text("\n".join(lines))
    print(f"\nSaved {tag} to {output_path}")


# %%
# ===========================================================
# ROBUSTNESS CHECK: LENGTH / COVERAGE / DIVERSITY / PUBLICATION YEAR
# ===========================================================

coverage_col = "perc_valid_scores_visual.mean"
robustness_datasets = {k: v for k, v in datasets.items() if k != "storyscope"}

log, lines = make_logger()

log("=" * 70)
log("ROBUSTNESS CHECK: LENGTH, COVERAGE, LEXICAL DIVERSITY, PUBLICATION YEAR")
log(f"Generated: {ts}")
log("=" * 70)

# ------------------------------------------------------------------
# 1. Within-Chicago: does token count predict visual score?
# ------------------------------------------------------------------
log("\n--- 1. Within-Chicago regression: n_tokens -> visual score ---\n")

chic_df = robustness_datasets["chicago"]
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
for name, df in robustness_datasets.items():
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

for name, df in robustness_datasets.items():
    rho, pval = stats.spearmanr(df["n_tokens"], df["avg_matched_visual.mean"])
    log(f"{name:25s}  rho={rho:+.3f}   p={pval:.4g}   n={len(df)}")

# ------------------------------------------------------------------
# 4. Coverage & lexical diversity: means/SDs, plus correlation with visual score
# ------------------------------------------------------------------
log("\n--- 4. Coverage & MSTTR: descriptive stats and correlation with visual score ---\n")

for name, df in robustness_datasets.items():
    log(f"\n{name}")
    log("-" * len(name))

    cov_mean, cov_sd = df[coverage_col].mean(), df[coverage_col].std()
    valid_msttr = df["msttr_lemmas"].notna()
    msttr_mean = df.loc[valid_msttr, "msttr_lemmas"].mean()
    msttr_sd = df.loc[valid_msttr, "msttr_lemmas"].std()

    log(f"  Coverage (perc_valid_scores_visual):  mean={cov_mean:.3f}  sd={cov_sd:.3f}  n={len(df)}")
    log(f"  MSTTR (lexical diversity):             mean={msttr_mean:.3f}  sd={msttr_sd:.3f}  n={valid_msttr.sum()} (of {len(df)})")

    rho_cov, p_cov = stats.spearmanr(df[coverage_col], df["avg_matched_visual.mean"])
    rho_msttr, p_msttr = stats.spearmanr(
        df.loc[valid_msttr, "msttr_lemmas"], df.loc[valid_msttr, "avg_matched_visual.mean"]
    )
    log(f"  Coverage vs. visual score:  rho={rho_cov:+.3f}  p={p_cov:.4g}")
    log(f"  MSTTR vs. visual score:     rho={rho_msttr:+.3f}  p={p_msttr:.4g}")

# ------------------------------------------------------------------
# 5. Publication year (Chicago only): does era confound coverage or visual score?
# ------------------------------------------------------------------
log("\n--- 5. Publication year (Chicago only): coverage/visual confound check ---\n")

valid_year = chic_df["year"].notna()
log(f"Year range: {chic_df['year'].min():.0f}--{chic_df['year'].max():.0f}")
log(f"N with valid year: {valid_year.sum()} (of {len(chic_df)})")

rho_cov_year, p_cov_year = stats.spearmanr(
    chic_df.loc[valid_year, "year"], chic_df.loc[valid_year, coverage_col]
)
log(f"\nCoverage vs. year:     rho={rho_cov_year:+.3f}  p={p_cov_year:.4g}")

rho_vis_year, p_vis_year = stats.spearmanr(
    chic_df.loc[valid_year, "year"], chic_df.loc[valid_year, "avg_matched_visual.mean"]
)
log(f"Visual score vs. year: rho={rho_vis_year:+.3f}  p={p_vis_year:.4g}")

model_year = smf.ols(
    'Q("avg_matched_visual.mean") ~ year + Q("perc_valid_scores_visual.mean")',
    data=chic_df.loc[valid_year]
).fit()
log(f"\nRegression: visual score ~ year + coverage")
log(f"  year:     beta = {model_year.params['year']:+.4e}   p = {model_year.pvalues['year']:.4g}")
log(f"  coverage: beta = {model_year.params['Q(\"perc_valid_scores_visual.mean\")']:+.4f}   "
    f"p = {model_year.pvalues['Q(\"perc_valid_scores_visual.mean\")']:.4g}")
log(f"  R-squared = {model_year.rsquared:.4f}, n = {int(model_year.nobs)}")

log("\n" + "=" * 70)
log("END OF ROBUSTNESS CHECK")
save_log(lines, OUT_DIR, ts, tag="robustness_check")


# %%
# ===========================================================
# FANDOM COHERENCE CHECK (descriptive: means/SDs by fandom)
# ===========================================================

fanfics = datasets["fanfics"]
print(fanfics["fandom_label"].value_counts())

log, lines = make_logger()

log("=" * 60)
log("FANDOM-LEVEL SENSORY PROFILE COMPARISON (within Fanfiction)")
log("=" * 60)

for fandom, group in fanfics.groupby("fandom_label"):
    log(f"\n{fandom} (n={len(group)})")
    log("-" * (len(fandom) + 10))
    for sense in sense_cols_prefixed:
        label = sense.replace(USE_WHAT, "").replace(".mean", "")
        log(f"  {label:15s}  mean={group[sense].mean():.3f}  sd={group[sense].std():.3f}")

save_log(lines, OUT_DIR, ts, tag="fandom_coherence_summary")

# visual check
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
axes = axes.flatten()

fandom_palette = sns.color_palette("husl", fanfics["fandom_label"].nunique())
fandom_colors = dict(zip(sorted(fanfics["fandom_label"].unique()), fandom_palette))

for i, sense in enumerate(senses):
    ax = axes[i]
    col = f'{USE_WHAT}{sense}.mean'
    for fandom, group in fanfics.groupby("fandom_label"):
        sns.kdeplot(data=group, x=col, label=fandom, color=fandom_colors[fandom], ax=ax, linewidth=1.5)
    ax.set_xlabel(sense.capitalize())
    ax.set_ylabel('Density' if i % 3 == 0 else '')

axes[0].legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / f"{ts}_fandom_kde.png", bbox_inches='tight')
plt.show()


# %%
# ===========================================================
# INTERNAL COHERENCE CHECK: Storyscope models & Fanfic fandoms
# ===========================================================

def check_internal_coherence(df, group_col, sense_cols, label, ts, out_dir, figs_dir,
                              random_state=42, run_classifier=True):
    log, lines = make_logger()

    log("=" * 70)
    log(f"INTERNAL COHERENCE CHECK: {label} (grouped by '{group_col}')")
    log("=" * 70)
    log(str(df[group_col].value_counts(dropna=False)))

    log("\n--- Levene's test (variance homogeneity per sense) ---")
    levene_results = {}
    for sense in sense_cols:
        s_label = sense.replace(USE_WHAT, "").replace(".mean", "")
        groups = [g[sense].dropna().values for _, g in df.groupby(group_col)]
        stat, p = levene(*groups)
        log(f"{s_label:15s}  Levene's W={stat:.2f}  p={p:.4g}")
        levene_results[s_label] = {"W": float(stat), "p": float(p)}

    slug = label.lower().replace(" ", "_")
    result = None

    if run_classifier:
        d = df.copy()
        sub_label_map = {name: i for i, name in enumerate(sorted(d[group_col].dropna().unique()))}
        d = d[d[group_col].notna()].copy()
        d["label"] = d[group_col].map(sub_label_map)

        missing_mask = d["author"].isna()
        if missing_mask.any():
            d.loc[missing_mask, "author"] = [f"missing_author_{i}" for i in range(missing_mask.sum())]
            log(f"Filled {missing_mask.sum()} missing authors with dummy ids")
        d["author"] = d["author"].astype("object")

        target_n = d.groupby("label").size().min()
        pieces = [group.sample(n=target_n, random_state=random_state) for _, group in d.groupby("label")]
        d_balanced = pd.concat(pieces, axis=0).reset_index(drop=True)

        result = run_classification(d_balanced, sense_cols, sub_label_map, random_state=random_state)

        log(f"\n{label} internal separability:")
        log(f"  accuracy = {np.mean(result['scores']['accuracy']):.4f} ± {np.std(result['scores']['accuracy']):.4f}")
        log(f"  macro_f1 = {np.mean(result['scores']['macro_f1']):.4f} ± {np.std(result['scores']['macro_f1']):.4f}")
        log(f"  roc_auc_ovr = {np.mean(result['scores']['roc_auc_ovr']):.4f} ± {np.std(result['scores']['roc_auc_ovr']):.4f}")

        chance = 1 / len(sub_label_map)
        log(f"  chance level (1/{len(sub_label_map)}) = {chance:.4f}")
        log(f"  accuracy above chance = {np.mean(result['scores']['accuracy']) - chance:+.4f}")

        log("\n--- Per-Class Performance (mean ± std) ---")
        for c, name in zip(result["class_labels"], result["class_display_names"]):
            log(f"\nClass {c} ({name}):")
            for metric in ["precision", "recall", "f1"]:
                key = f"{c}_{metric}"
                vals = result["per_class_scores"][key]
                log(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        class_display_names = result["class_display_names"]
        n_classes = len(class_display_names)
        fig_size = max(3, n_classes * 0.9)
        plt.figure(figsize=(fig_size, fig_size), dpi=500)
        sns.heatmap(result["cm"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_display_names, yticklabels=class_display_names)
        plt.xlabel(r"$\bf{Predicted}$")
        plt.ylabel(r"$\bf{True}$")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(figs_dir / f"{ts}_internal_coherence_{slug}_confusion_matrix.png")
        plt.show()

        json_out = {
            "timestamp": ts,
            "label": label,
            "group_col": group_col,
            "levene": levene_results,
            "chance_level": chance,
            "overall_metrics": {
                m: {"mean": float(np.mean(result["scores"][m])), "std": float(np.std(result["scores"][m]))}
                for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]
            },
            "per_class_metrics": {
                name: {
                    m: {
                        "mean": float(np.mean(result["per_class_scores"][f"{c}_{m}"])),
                        "std": float(np.std(result["per_class_scores"][f"{c}_{m}"])),
                    }
                    for m in ["precision", "recall", "f1"]
                }
                for c, name in zip(result["class_labels"], result["class_display_names"])
            },
            "confusion_matrix": result["cm"].tolist(),
            "label_map": sub_label_map,
        }
        (out_dir / f"{ts}_internal_coherence_{slug}_results.json").write_text(json.dumps(json_out, indent=2))
        log(f"Saved structured results to {ts}_internal_coherence_{slug}_results.json")

    log("=" * 70)
    save_log(lines, out_dir, ts, tag=f"internal_coherence_{slug}_summary")

    return result


storyscope_with_model = pd.concat(
    [datasets[k].assign(model=k.replace("storyscope_", "")) for k in storyscope_keys],
    axis=0, ignore_index=True,
)

storyscope_result = check_internal_coherence(
    storyscope_with_model, group_col="model", sense_cols=sense_cols_prefixed, label="Storyscope models",
    ts=ts, out_dir=OUT_DIR, figs_dir=FIGS,
)

fandom_result = check_internal_coherence(
    datasets["fanfics"], group_col="fandom_label", sense_cols=sense_cols_prefixed, label="Fanfic fandoms",
    ts=ts, out_dir=OUT_DIR, figs_dir=FIGS,
)


# %%
# ===========================================================
# EFFECT SIZES: BETWEEN-CORPUS DIFFERENCES (Cohen's d + Levene's)
# ===========================================================

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    m1, m2 = group1.mean(), group2.mean()
    sd1, sd2 = group1.std(), group2.std()
    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled_sd


log, lines = make_logger()

log("=" * 70)
log("EFFECT SIZES: BETWEEN-CORPUS DIFFERENCES")
log(f"Generated: {ts}")
log("=" * 70)

generated_pooled = pd.concat(
    [datasets["simplestories"]] + [datasets[k] for k in storyscope_keys],
    axis=0, ignore_index=True,
)
storyscope_combined_for_levene = pd.concat(
    [datasets[k] for k in storyscope_keys], axis=0, ignore_index=True
)

comparisons = [
    ("Chicago vs Fanfiction", datasets["chicago"], datasets["fanfics"]),
    ("Chicago vs Generated (pooled)", datasets["chicago"], generated_pooled),
    ("Fanfiction vs Generated (pooled)", datasets["fanfics"], generated_pooled),
]

for i, (title, g1, g2) in enumerate(comparisons, 1):
    log(f"\n--- {i}. Cohen's d: {title}, per sense ---\n")
    for sense in sense_cols_prefixed:
        label = sense.replace(USE_WHAT, "").replace(".mean", "")
        d = cohens_d(g1[sense], g2[sense])
        log(f"{label:15s}  d={d:+.3f}")

log(f"\n--- 4. Levene's test: Chicago vs Storyscope (combined), per sense ---\n")
for sense in sense_cols_prefixed:
    label = sense.replace(USE_WHAT, "").replace(".mean", "")
    stat, p = levene(datasets["chicago"][sense], storyscope_combined_for_levene[sense])
    log(f"{label:15s}  Levene's W={stat:.2f}  p={p:.4g}")

log("\n" + "=" * 70)
log("END OF EFFECT SIZE CHECKS")
save_log(lines, OUT_DIR, ts, tag="effect_sizes")
# %%
