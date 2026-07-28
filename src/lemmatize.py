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

FILENAME = "fanfics_mia"
SAVENAME = FILENAME + "_lemmatized.json"

# # load a file / dataset
# ds = load_dataset("jjrussell10/storyscope")
# df = pd.DataFrame(ds["train"])
# rename
#df = df.rename(columns={"story": "text"})
# sample subset of the data for testing
#df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# OR file

df = pd.read_csv(DATA_PATH / "data_subset_chr27.csv", sep=",")#/ "2026-07-09_fanfics_cleaned.json", orient='records', lines=True)
print(f"len of df: {len(df)}")
print(df.columns)
df.head()


# %%

# for storyscope, we need to change so that e.g. col "story_claude" and "story_gemini"
# become rows "text", with their "prompt_id"
dfs = []
for col in df.columns:
    if col.startswith("story_"):
        # get the prompt_id from the column name
        source = col.split("_")[1]
        # get the "prompt_id" column
        prompt_id = df["prompt_id"]
        # create a new dataframe with the text and source + prompt_id
        new_df = pd.DataFrame({
            "text": df[col],
            "source": source,
            "prompt_id": prompt_id,
            "human_author": df["human_author"]
        })
        print(f"len of new_df for {col}: {len(new_df)}")
        dfs.append(new_df)

df = pd.concat(dfs, ignore_index=True)

df["work_id"] = df["prompt_id"].astype(str) + "_" + df["source"]

df = df[["work_id", "text", "human_author"]].copy()
print(f"len of df: {len(df)}")
df.head()

# # %%

# # BNC
# bnc = pd.read_csv(DATA_PATH / "bnc_serge_sharoff.tsv", sep="\t", names=["text", "raw", "adjusted", "clipped", "total_doc"])
# bnc.head()

# df = bnc.copy()

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
