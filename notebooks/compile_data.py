# %%
"""
build_dataset_checkpoint.py

Loads, merges, and labels all raw scored datasets, then saves a checkpoint
(compressed JSON-lines, one file per dataset) that the main analysis script
can load directly -- skipping the loading/merging/labeling step on every run.

Run this script whenever the underlying raw data changes (e.g. new fanfic
fandoms added, more Storyscope splits pulled in). Otherwise, the main script
just loads the checkpoint.
"""

import json
import time
from pathlib import Path

import pandas as pd
from scipy.stats import entropy

ts = time.strftime("%Y-%m-%d")
print(f"Timestamp: {ts}")

CWD = Path(__file__).parent
DATA_PATH = CWD.parent / "data" / "lemmatized_data" / "scored_data"
CHECKPOINT_DIR = CWD.parent / "data" / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

log(f"{'='*70}\nBUILD DATASET CHECKPOINT\nGenerated: {ts}\n{'='*70}")

# %%
### Load all raw datasets ###

datasets = {}
for path in sorted(Path(DATA_PATH).glob("*.json")):
    name = path.stem.replace("_with_scores", "").replace("_lemmatized", "")
    datasets[name] = pd.read_json(path, orient="records", lines=True)
    log(f"Loaded {name}: {len(datasets[name])} entries")

log(f"\nDatasets loaded: {sorted(datasets.keys())}")

# %%

### Give SimpleStories a work_id (it doesn't have one natively) ###

if "work_id" not in datasets["simplestories"].columns:
    datasets["simplestories"] = datasets["simplestories"].reset_index(drop=True)
    datasets["simplestories"]["work_id"] = "simplestories_" + datasets["simplestories"].index.astype(str)
    log(f"Added work_id to simplestories (e.g. {datasets['simplestories']['work_id'].iloc[0]})")

### Merge extra Storyscope validation/test rows into the main storyscope dataset ###

if "testsetstoryscope" in datasets:
    log(f"\n--- Merging testsetstoryscope into storyscope ---")
    log(f"storyscope before merge: {len(datasets['storyscope'])} rows")
    log(f"testsetstoryscope: {len(datasets['testsetstoryscope'])} rows")

    shared_cols = set(datasets["storyscope"].columns) & set(datasets["testsetstoryscope"].columns)
    only_in_storyscope = set(datasets["storyscope"].columns) - shared_cols
    only_in_testset = set(datasets["testsetstoryscope"].columns) - shared_cols
    log(f"Columns only in storyscope: {only_in_storyscope}")
    log(f"Columns only in testsetstoryscope: {only_in_testset}")

    before_n = len(datasets["storyscope"]) + len(datasets["testsetstoryscope"])
    datasets["storyscope"] = pd.concat(
        [datasets["storyscope"], datasets["testsetstoryscope"]], axis=0, ignore_index=True
    )
    datasets["storyscope"] = datasets["storyscope"].drop_duplicates(subset="work_id").reset_index(drop=True)
    dropped = before_n - len(datasets["storyscope"])
    del datasets["testsetstoryscope"]

    log(f"storyscope after merge: {len(datasets['storyscope'])} rows ({dropped} duplicate work_ids dropped)")
else:
    log("\nNo 'testsetstoryscope' dataset found -- skipping merge.")

# %%
### Build author labels ###

log(f"\n--- Building author labels ---")

datasets["simplestories"]["author"] = (
    datasets["simplestories"]["persona"] + "_" + datasets["simplestories"]["style"]
)

datasets["storyscope"]["model"] = datasets["storyscope"]["work_id"].str.extract(r"_([a-zA-Z]+)$")
datasets["storyscope"]["author"] = datasets["storyscope"]["model"] + "_" + datasets["storyscope"]["human_author"]
log(str(datasets["storyscope"]["model"].value_counts(dropna=False)))

# --- fanfiction: tag each source with its fandom, then merge into one dataset ---
datasets["fanfics"]["fandom_label"] = "AGRL"

datasets["fanfics_mia"]["author"] = datasets["fanfics_mia"]["work_author"]
if "fandom_label" not in datasets["fanfics_mia"].columns:
    datasets["fanfics_mia"]["fandom_label"] = datasets["fanfics_mia"]["fandom"]

shared_cols = set(datasets["fanfics"].columns) & set(datasets["fanfics_mia"].columns)
only_in_agrl = set(datasets["fanfics"].columns) - shared_cols
only_in_mia = set(datasets["fanfics_mia"].columns) - shared_cols
log(f"Columns only in fanfics (AGRL): {only_in_agrl}")
log(f"Columns only in fanfics_mia: {only_in_mia}")

datasets["fanfics"] = pd.concat(
    [datasets["fanfics"], datasets["fanfics_mia"]], axis=0, ignore_index=True
)
del datasets["fanfics_mia"]

log(f"Combined fanfics dataset: {len(datasets['fanfics'])} entries "
    f"({datasets['fanfics']['fandom_label'].value_counts().to_dict()})")

# %%
### Split storyscope into per-model datasets (combined "storyscope" stays too) ###

log(f"\n--- Splitting storyscope by model ---")

for model, group in datasets["storyscope"].groupby("model"):
    datasets[f"storyscope_{model}"] = group.drop(columns="model").reset_index(drop=True)

storyscope_keys = [k for k in datasets if k.startswith("storyscope_")]
log(f"Storyscope model keys: {storyscope_keys}")
log("Per-model sizes (untrimmed):")
for k in storyscope_keys:
    log(f"  {k}: {len(datasets[k])}")

datasets["storyscope"] = datasets["storyscope"].drop(columns="model")

log("\nAuthors with >10 stories, per source:")
for key in ["simplestories", "storyscope", *storyscope_keys]:
    vc = datasets[key]["author"].value_counts()
    log(f"\n{key}:")
    log(str(vc[vc > 10]))

# %%
### Add Chicago metadata ###

log(f"\n--- Adding Chicago metadata ---")

datasets["chicago"] = datasets["chicago"].drop(columns=["text"])

meta = pd.read_excel(CWD.parent / "data" / "meta" / "CHICAGO_MEASURES_MARCH24.xlsx")
meta = meta[["BOOK_ID", "AUTH_FIRST", "AUTH_LAST", "WORDCOUNT", "PUBL_DATE"]]
meta.columns = ["work_id", "author_first", "author_last", "text_length", "year"]
meta["author"] = meta["author_first"] + " " + meta["author_last"]
meta.drop(columns=["author_first", "author_last"], inplace=True)

chic = datasets["chicago"].merge(meta, how="left", on="work_id")
before_n = len(chic)
chic = chic.drop_duplicates(subset="work_id")
log(f"Chicago: {before_n} -> {len(chic)} after dropping duplicate work_ids")

chic.rename(columns={
    col: "avg_matched_" + col.replace("_mean", ".mean")
    for col in chic.columns
    if col.endswith("_mean") and any(
        sense in col for sense in
        ["auditory", "gustatory", "haptic", "interoceptive", "olfactory", "visual"]
    )
}, inplace=True)

datasets["chicago"] = chic

# %%
### Sense columns and derived metrics ###

log(f"\n--- Computing sense entropy ---")

sense_cols = ["auditory.mean", "gustatory.mean", "olfactory.mean", "haptic.mean", "visual.mean", "interoceptive.mean"]
USE_WHAT = "avg_matched_"
sense_cols_prefixed = [USE_WHAT + col for col in sense_cols]


def add_entropy(df, sense_cols_prefixed):
    df[f"{USE_WHAT}sense_sum"] = df[sense_cols_prefixed].sum(axis=1)
    for sense in sense_cols_prefixed:
        colname = sense.replace(".mean", "_percent")
        df[colname] = df[sense] / df[f"{USE_WHAT}sense_sum"]
    df["sense_entropy"] = df[[c for c in df.columns if c.endswith("_percent")]].apply(
        lambda x: entropy(x, base=2), axis=1
    )
    return df


for name, df in datasets.items():
    datasets[name] = add_entropy(df, sense_cols_prefixed)

log("Entropy computed for all datasets.")


# %%
### Save checkpoint: split into a small "scores" checkpoint (git-friendly) ###
### and a large "text" checkpoint (kept local only, not committed) ###

log(f"\n--- Saving checkpoints ---")

SCORES_DIR = CHECKPOINT_DIR / "scores"
TEXT_DIR = CHECKPOINT_DIR / "text_local"  # not committed to git
SCORES_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

total_scores_bytes = 0
total_text_bytes = 0

for name, df in datasets.items():
    has_text = "lemmatized_text" in df.columns

    # --- scores checkpoint: everything except the raw token lists ---
    scores_df = df.drop(columns=["lemmatized_text"]) if has_text else df
    scores_path = SCORES_DIR / f"{name}.json.gz"
    scores_df.to_json(
        scores_path, orient="records", lines=True, force_ascii=False,
        compression={"method": "gzip", "compresslevel": 1},
    )
    size_mb = scores_path.stat().st_size / (1024 * 1024)
    total_scores_bytes += scores_path.stat().st_size
    log(f"  [scores] {name:20s}  {len(scores_df):>7} rows  ->  {scores_path.name}  ({size_mb:.2f} MB)")

    # --- text checkpoint: work_id + lemmatized_text only, for re-joining later ---
    if has_text:
        text_df = df[["work_id", "lemmatized_text"]].copy()
        text_df["lemmatized_text"] = text_df["lemmatized_text"].apply(
            lambda toks: " ".join(toks) if isinstance(toks, list) else toks
        )
        text_path = TEXT_DIR / f"{name}_text.json.gz"
        text_df.to_json(
            text_path, orient="records", lines=True, force_ascii=False,
            compression={"method": "gzip", "compresslevel": 1},
        )
        size_mb = text_path.stat().st_size / (1024 * 1024)
        total_text_bytes += text_path.stat().st_size
        log(f"  [text]   {name:20s}  {len(text_df):>7} rows  ->  {text_path.name}  ({size_mb:.2f} MB)")

log(f"\nScores checkpoint total: {total_scores_bytes / (1024*1024):.1f} MB  (this is what goes to GitHub)")
log(f"Text checkpoint total:   {total_text_bytes / (1024*1024):.1f} MB  (local only -- do not commit)")

config = {"sense_cols": sense_cols, "USE_WHAT": USE_WHAT, "sense_cols_prefixed": sense_cols_prefixed,
          "storyscope_keys": storyscope_keys}
(CHECKPOINT_DIR / "checkpoint_config.json").write_text(json.dumps(config, indent=2))
log(f"Saved checkpoint config to checkpoint_config.json")

log(f"\n{'='*70}\nDONE\n{'='*70}")

(CHECKPOINT_DIR / f"{ts}_build_log.txt").write_text("\n".join(lines))
print(f"\nSaved build log to {CHECKPOINT_DIR / f'{ts}_build_log.txt'}")
# %%
