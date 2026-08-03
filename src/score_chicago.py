# %%
import pandas as pd
from pathlib import Path
from lexical_diversity import lex_div as ld

# %%

CWD = Path(__file__).parent
DATA_PATH = CWD.parent / "data" / "lemmatized_data"
RESOURCES_PATH = CWD.parent / "resources"
CHICAGO_LEMMA_DIR = CWD.parent.parent / "Chicago_lemmatized"  # two dirs up
print(CHICAGO_LEMMA_DIR)
# %%
norms = pd.read_csv(RESOURCES_PATH / "cleaned_sensorimotor_norms.csv")

OUTPUT_PATH = DATA_PATH / "scored_data"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %%
# --- Load Chicago from its per-file lemmatized directory ---

def clean_tokens(toks):
    """Lowercase and drop non-alphabetic tokens (fixes the casing/punctuation bug)."""
    return [t.lower() for t in toks if isinstance(t, str) and t.isalpha()]


rows = []
chicago_files = sorted(CHICAGO_LEMMA_DIR.glob("*.txt"))  # ADJUST extension if not .txt
print(f"Found {len(chicago_files)} Chicago lemma files")

for path in chicago_files:
    work_id = path.stem  # filename (without extension) used as file id -- adjust if your ids are formatted differently
    with open(path, encoding="utf-8") as f:
        raw_tokens = f.read().split()  # ADJUST if tokens are stored one-per-line or as JSON instead of whitespace-separated
    tokens = clean_tokens(raw_tokens)
    rows.append({"work_id": work_id, "lemmatized_text": tokens})

data = pd.DataFrame(rows)
print(f"Loaded {len(data)} Chicago texts")
data.head()

# %%
def get_dict_scores_all_modalities(texts, dictionary, norms_cols):
    """
    Single pass over each text's tokens, scoring all six modalities at once,
    instead of looping over every text 6 separate times (once per modality).
    """
    results = {col: {"total": [], "normalized": [], "avg_matched": [], "avg_matched_sd": [],
                      "n_valid": []} for col in norms_cols}
    n_tokens_list = []
    unique_vs_all_lemmas = []
    msttr_lemmas = []

    for text in texts:
        n_tokens_list.append(len(text))

        # collect valid scores per modality in one pass over tokens
        valid_scores = {col: [] for col in norms_cols}
        for token in text:
            entry = dictionary.get(token)
            if entry is not None:
                for col in norms_cols:
                    val = entry.get(col)
                    if val is not None:
                        valid_scores[col].append(val)

        for col in norms_cols:
            vs = valid_scores[col]
            total = sum(vs)
            results[col]["total"].append(total)
            results[col]["normalized"].append(total / len(text) if text else 0)
            results[col]["avg_matched"].append(total / len(vs) if vs else 0)
            results[col]["avg_matched_sd"].append(pd.Series(vs).std() if vs else 0)
            results[col]["n_valid"].append(len(vs))

        # computed ONCE per text now, not 6 times
        unique_lemmas = set(text)
        unique_vs_all_lemmas.append(len(unique_lemmas) / len(text) if text else 0)
        msttr = ld.msttr(text, window_length=100) if len(text) >= 100 else None
        msttr_lemmas.append(msttr)

    return results, n_tokens_list, unique_vs_all_lemmas, msttr_lemmas
    
# %%
from multiprocessing import Pool, cpu_count
import numpy as np

norms_cols = ['auditory.mean', 'gustatory.mean', 'haptic.mean',
              'interoceptive.mean', 'olfactory.mean', 'visual.mean']

norms['word'] = norms['word'].str.lower()
norms_lookup = norms.set_index('word')[norms_cols].to_dict(orient='index')

# %%
def chunk_list(lst, n_chunks):
    """Split a plain Python list into n_chunks roughly-equal pieces, without numpy."""
    k, m = divmod(len(lst), n_chunks)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def parallel_score(texts, dictionary, cols, n_workers=None):
    n_workers = n_workers or cpu_count()
    chunks = chunk_list(texts, n_workers)
    args = [(chunk, dictionary, cols) for chunk in chunks]

    with Pool(n_workers) as pool:
        chunk_results = pool.map(score_chunk, args)

    merged_results = {col: {"total": [], "normalized": [], "avg_matched": [], "avg_matched_sd": [], "n_valid": []}
                       for col in cols}
    merged_n_tokens, merged_unique, merged_msttr = [], [], []

    for results, n_tokens_list, unique_list, msttr_list in chunk_results:
        for col in cols:
            for key in merged_results[col]:
                merged_results[col][key].extend(results[col][key])
        merged_n_tokens.extend(n_tokens_list)
        merged_unique.extend(unique_list)
        merged_msttr.extend(msttr_list)

    return merged_results, merged_n_tokens, merged_unique, merged_msttr

# %%
print(f"Scoring {len(data)} texts using {cpu_count()} workers...")
results, n_tokens, unique_vs_all_lemmas, msttr_lemmas = parallel_score(
    data['lemmatized_text'].tolist(), norms_lookup, norms_cols, n_workers=cpu_count()
)

data['n_tokens'] = n_tokens
data['unique_vs_all_lemmas'] = unique_vs_all_lemmas
data['msttr_lemmas'] = msttr_lemmas

for col in norms_cols:
    data[f'total_{col}'] = results[col]["total"]
    data[f'normalized_{col}'] = results[col]["normalized"]
    data[f'avg_matched_{col}'] = results[col]["avg_matched"]
    data[f'avg_matched_{col}_sd'] = results[col]["avg_matched_sd"]
    data[f'n_valid_scores_{col}'] = results[col]["n_valid"]
    data[f'perc_valid_scores_{col}'] = data[f'n_valid_scores_{col}'] / data['n_tokens']

data.head()

# %%
savename = OUTPUT_PATH / "chicago_lemmatized_with_scores_new.json"
data.to_json(savename, orient='records', lines=True)
print(f"Saved {len(data)} rows to {savename}")
# %%
