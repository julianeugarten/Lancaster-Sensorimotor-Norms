# %%

from datasets import load_dataset
import spacy
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# %%

CWD = Path.cwd()

DATA_PATH = CWD.parent / "data"
print(DATA_PATH)
OUT_DIR= DATA_PATH / "lemmatized_data"
mkdir = OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =========== Adjust below =============

FILENAME = "storyscope_testset"
SAVENAME = FILENAME + "_lemmatized.json"

# # load a file / dataset
ds = load_dataset("jjrussell10/storyscope")
print(ds)  # check split names/sizes

# skip "train" since it's already processed and saved
splits_to_load = [s for s in ds.keys() if s != "train"]
print(f"Loading splits: {splits_to_load}")

df = pd.concat([pd.DataFrame(ds[split]) for split in splits_to_load], ignore_index=True)
print(f"Combined new splits: {len(df)} rows")

# =========== Adjust above =============

# %%

# for storyscope, we need to change so that e.g. col "story_claude" and "story_gemini"
# become rows "text", with their "prompt_id"
dfs = []
for col in df.columns:
    if col.startswith("story_"):
        source = col.split("_")[1]
        prompt_id = df["prompt_id"]
        new_df = pd.DataFrame({
            "text": df[col],
            "source": source,
            "prompt_id": prompt_id,
            "human_author": df["human_author"],
        })
        print(f"len of new_df for {col}: {len(new_df)}")
        dfs.append(new_df)

df = pd.concat(dfs, ignore_index=True)

df["work_id"] = df["prompt_id"].astype(str) + "_" + df["source"]

df = df[["work_id", "text", "human_author", "source"]].copy()
print(f"len of df: {len(df)}")

# --- ensure exactly 10,000 per model source, dropping duplicates and trimming excess ---
print("\nBefore dedup/trim:")
print(df["source"].value_counts())

# drop exact duplicate work_ids first, in case splits overlapped
df = df.drop_duplicates(subset="work_id").reset_index(drop=True)

print("\nAfter dedup/trim:")
print(df["source"].value_counts())

df.head()

# =========== Adjust above =============




# %%


# lemmatize the texts using spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
nlp.max_length = 10945772 
spacy.require_cpu()

def lemmatize_docs(texts):
    lemmas_list = []
    for doc in tqdm(nlp.pipe(texts, batch_size=50), total=len(texts)):
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        lemmas_list.append(lemmas)
    return lemmas_list

# %%

texts = df['text'].dropna().tolist()
print(len(texts))

# apply lemmatization
lemmatized = lemmatize_docs(texts)

# %%

# Keep only rows with non-NaN text (matches the lemmatized results)
df = df.dropna(subset=['text']).reset_index(drop=True)

# Now lengths match - assignment will work
df['lemmatized_text'] = lemmatized

# okay, let's drop the original text column to save space
df = df.drop(columns=['text'])

# save the cleaned metadata with texts and lemmatizations
df.to_json(OUT_DIR / SAVENAME, orient='records', force_ascii=False)


# %%
