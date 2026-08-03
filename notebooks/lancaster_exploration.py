# %%


import json
import time
import numpy as np
from pathlib import Path
import openpyxl
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ts = time.strftime("%Y-%m-%d")
print(f"Timestamp: {ts}")

CWD = Path(__file__).parent
DATA_PATH = CWD.parent / "data" / "checkpoint"
FIGS = CWD.parent / "figs"
OUT_DIR = CWD / "OUT_DIR"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Get lancaster
RESOURCES_PATH = CWD.parent / "resources"
norms = pd.read_csv(RESOURCES_PATH / "cleaned_sensorimotor_norms.csv")

norms_cols = ['auditory.mean', 'gustatory.mean', 'haptic.mean',
       'interoceptive.mean', 'olfactory.mean', 'visual.mean'] #, 'foot_leg.mean', 'hand_arm.mean', 'head.mean', 'mouth.mean', 'torso.mean']

norms_lookup = norms.set_index('word')[norms_cols].to_dict(orient='index')


# %%
### Load checkpoint (scores always; lemmatized text only if available locally) ###

SCORES_DIR = DATA_PATH / "scores"
TEXT_DIR = DATA_PATH / "text_local"

config = json.loads((DATA_PATH / "checkpoint_config.json").read_text())
sense_cols = config["sense_cols"]
USE_WHAT = config["USE_WHAT"]
sense_cols_prefixed = config["sense_cols_prefixed"]
storyscope_keys = config["storyscope_keys"]

HAS_TEXT = TEXT_DIR.exists() and any(TEXT_DIR.glob("*.json.gz"))
print(f"Text checkpoint available: {HAS_TEXT}")

datasets = {}
for path in sorted(SCORES_DIR.glob("*.json.gz")):
    name = path.stem.replace(".json", "")
    d = pd.read_json(path, orient="records", lines=True, compression="gzip")

    if HAS_TEXT:
        text_path = TEXT_DIR / f"{name}_text.json.gz"
        if text_path.exists():
            text_df = pd.read_json(text_path, orient="records", lines=True, compression="gzip")
            text_df["lemmatized_text"] = text_df["lemmatized_text"].str.split(" ")
            d = d.merge(text_df, on="work_id", how="left")

    datasets[name] = d
    print(f"Loaded {name}: {len(datasets[name])} rows"f"{' (with text)' if 'lemmatized_text' in d.columns else ''}")

# %%

### REMOVE VECTOR IF 0 ACROSS ALL VALUES IN SENSE VECTOR ####

# drop rows with zero total sense score (degenerate/near-empty texts)
for ds in list(datasets.keys()):
    zero_mask = datasets[ds][sense_cols_prefixed].sum(axis=1) == 0
    print(f"N zeroes {ds}: {zero_mask.sum()} out of {len(zero_mask)}")
    datasets[ds] = datasets[ds][~zero_mask].reset_index(drop=True)
    print(f"After filtering: {len(datasets[ds])} (dropped {zero_mask.sum()})")


# %%

# Check out 

# pool token counts across all datasets (needs lemmatized_text -- local text checkpoint)
word_counts = Counter()
for name, df in datasets.items():
    for toks in df['lemmatized_text']:
        word_counts.update(toks)

print(f"Total unique word types in pooled corpora: {len(word_counts)}")

sense_cols = ['auditory.mean', 'gustatory.mean', 'olfactory.mean', 'haptic.mean', 'visual.mean', 'interoceptive.mean']

word_counts
# %%

# Building the per-modality word lists
top_k = 15
percentile_cutoff = 90  # only consider words in the top quartile for this specific sense
# 15 words per modality panel, drawn only from the top 10% of words by score on that specific sense

# out of the top 10% of words for that modality, what are the most frequent 15?
rows = []
for sense in sense_cols:
    scores = pd.Series({w: v[sense] for w, v in norms_lookup.items() if v.get(sense) is not None})
    threshold = scores.quantile(percentile_cutoff / 100)
    high_scoring_words = set(scores[scores >= threshold].index)

    candidates = [(w, word_counts[w], scores[w]) for w in high_scoring_words if w in word_counts]
    candidates.sort(key=lambda x: -x[1])  # now sort by frequency, WITHIN the high-scoring set

    label = sense.replace('.mean', '')
    for w, freq, score in candidates[:top_k]:
        rows.append({'modality': label, 'word': w, 'freq': freq, 'score': score})

top_df = pd.DataFrame(rows)
print(top_df)


fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

sns.set_style("whitegrid")

for i, sense in enumerate(sense_cols):
    ax = axes[i]
    label = sense.replace('.mean', '')
    sub = top_df[top_df['modality'] == label].sort_values('freq')
    ax.barh(sub['word'], sub['freq'], color='steelblue')
    ax.set_title(label.capitalize())
    ax.set_xlabel('Corpus frequency')

plt.tight_layout()
plt.savefig(FIGS / f"{ts}_top_words_per_modality.png", dpi=300)
plt.show()
# %%

top_n = 100

sorted_items = sorted(word_counts.items(), key=lambda x: -x[1])
top_words = sorted_items[:top_n]

words = [w for w, _ in top_words]
freqs = [c for _, c in top_words]
in_lancaster = [w in norms_lookup for w in words]

colors = ['crimson' if is_lanc else 'lightgray' for is_lanc in in_lancaster]

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(words)), freqs, color=colors)
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words, rotation=60, ha='right')
ax.set_ylabel('Corpus frequency')

# legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='crimson', label='In Lancaster'),
    Patch(facecolor='lightgray', label='Not in Lancaster'),
]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(FIGS / f"{ts}_top_words_lancaster_coverage.png", dpi=300)
plt.show()
# %%
sense = 'visual.mean'
top_n = 100

sorted_items = sorted(word_counts.items(), key=lambda x: -x[1])[:top_n]
words = [w for w, _ in sorted_items]
freqs = [c for _, c in sorted_items]
scores = [norms_lookup.get(w, {}).get(sense, 0) for w in words]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(words)), freqs, color=plt.cm.Greys(np.array(scores) / max(scores)))
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words, rotation=60, ha='right')
ax.set_ylabel('Corpus frequency')
ax.set_title(f'Top {top_n} frequent words, shaded by {sense.replace(".mean","")} score')

sm = plt.cm.ScalarMappable(cmap='Greys', norm=plt.Normalize(0, max(scores)))
plt.colorbar(sm, ax=ax, label=f'{sense.replace(".mean","")} score')

plt.tight_layout()
plt.savefig(FIGS / f"{ts}_top_words_shaded_by_{sense.replace('.mean','')}.png", dpi=300)
plt.show()

# %%

top_n = 2000
sorted_items = sorted(word_counts.items(), key=lambda x: -x[1])[:top_n]
missing = [(w, c) for w, c in sorted_items if w not in norms_lookup]
print(f"{len(missing)} of top {top_n} words are NOT in Lancaster:")
for w, c in missing:
    print(f"  {w:20s} freq={c}")
# %%

