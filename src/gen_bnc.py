# %%
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

CWD = Path.cwd()
DATA_PATH = CWD.parent / "data" / "lemmatized_data"
SCORED_DATA_PATH = DATA_PATH / "scored_data"
RESOURCES_PATH = CWD.parent / "resources"

norms = pd.read_csv(RESOURCES_PATH / "cleaned_sensorimotor_norms.csv")

SENSES = ["auditory", "gustatory", "olfactory", "haptic", "visual", "interoceptive"]
sense_cols_prefixed = [f"avg_matched_{s}.mean" for s in SENSES]

# %%
# --- Lancaster lookup: one flat dict per sense ---
lancaster_lookup = {}
for sense in SENSES:
    col = f"{sense}.mean"
    lancaster_lookup[sense] = norms.dropna(subset=[col]).set_index('word')[col].to_dict()

# %%
# --- Load and clean BNC frequency list ---
bnc = pd.read_json(DATA_PATH / "bnc_lemmatized.json", orient='records')
bnc = bnc.explode('lemmatized_text').reset_index(drop=True)

# sanity check: whelk/burstiness pattern (diagnostic only, not used downstream)
bnc["raw_per_doc"] = bnc["raw"] / bnc["total_doc"]
print(bnc.sort_values("raw_per_doc", ascending=False).head(20))

bnc = bnc.rename(columns={"lemmatized_text": "word"})
bnc["word"] = bnc["word"].astype(str).str.lower().str.strip()

junk_tokens = {"amp", "apo", "quot", "bquo", "equo"}  # expand after eyeballing more top rows
bnc = bnc[~bnc["word"].isin(junk_tokens)]
bnc = bnc.drop_duplicates(subset="word")
# keep as is


# %%
# --- Build frequency-weighted pools per sense, using ADJUSTED (robust) frequency ---
pools = {}
for sense in SENSES:
    pool = bnc[bnc["word"].isin(lancaster_lookup[sense].keys())].copy()
    pool = pool[pool["adjusted"] > 0]
    pool = pool.drop_duplicates(subset="word")
    pool["weight"] = pool["adjusted"] / pool["adjusted"].sum()
    pools[sense] = pool
    print(f"{sense}: pool size = {len(pool)}")

# %%
# --- Pool coverage / sanity check ---
for sense in SENSES:
    pool = pools[sense]
    n_lancaster = len(lancaster_lookup[sense])
    n_pool = len(pool)
    coverage_pct = n_pool / n_lancaster * 100
    eff_n = 1 / (pool["weight"]**2).sum()  # inverse Simpson index
    top5 = pool.sort_values("weight", ascending=False).head(5)[["word", "adjusted", "weight"]]

    print(f"\n=== {sense} ===")
    print(f"Lancaster words with a score: {n_lancaster}")
    print(f"Words found in BNC pool:      {n_pool} ({coverage_pct:.1f}% coverage)")
    print(f"Effective sample size (approx): {eff_n:.0f}")
    print("Top 5 highest-weight words:")
    print(top5.to_string(index=False))

# %%
# --- Null score generator: per-sense word counts, drawn from frequency-weighted pool ---
def generate_null_text_scores_for_row(row, pools, lancaster_lookup, fixed_n=None):
    """
    fixed_n=None -> length-matched: uses row's own n_valid_scores_{sense} per sense
    fixed_n=int  -> length-equalized: same word count for every sense/text
    """
    scores = {}
    for sense in SENSES:
        n_words = fixed_n if fixed_n is not None else int(row[f"n_valid_scores_{sense}.mean"])
        pool_df = pools[sense]
        words = pool_df["word"].values
        weights = pool_df["weight"].values
        sample = np.random.choice(words, size=max(n_words, 0), replace=True, p=weights)
        matched_scores = [lancaster_lookup[sense][w] for w in sample]
        scores[f"avg_matched_{sense}.mean"] = np.mean(matched_scores) if matched_scores else np.nan
    return scores

# %%
# --- Load the real classification dataset (must be saved from your classification script) ---
together = pd.read_pickle(DATA_PATH / "together_classification_set.pkl")  # adjust filename as needed

n_valid_cols = [f"n_valid_scores_{s}.mean" for s in SENSES]
missing = [c for c in n_valid_cols if c not in together.columns]
if missing:
    raise ValueError(f"Missing expected columns in `together`: {missing}. "
                      f"Re-run scoring script with per-sense n_valid_scores_{{sense}} columns and re-save `together`.")

# %%
# --- Variant 1: length-matched null (each null text matches its own real text's per-sense length) ---
null_rows_matched = []
for idx, row in together.iterrows():
    null_scores = generate_null_text_scores_for_row(row, pools, lancaster_lookup, fixed_n=None)
    null_scores["label"] = row["label"]