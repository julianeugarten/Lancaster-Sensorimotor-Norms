# %%

from datasets import load_dataset
import spacy
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# %%

CWD = Path.cwd()
DATA_PATH = CWD.parent / "data"
OUT_DIR= DATA_PATH / "lemmatized_data"
mkdir = OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========== Adjust below =============

FILENAME = "fanfics"
SAVENAME = FILENAME + "_lemmatized.json"

# # load a file / dataset
# ds = load_dataset("SimpleStories/SimpleStories")
# df = pd.DataFrame(ds["train"])
# # rename
# df = df.rename(columns={"story": "text"})
# # sample subset of the data for testing
# df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# OR file

df = pd.read_json(DATA_PATH / "2026-07-09_fanfics_cleaned.json", orient='records', lines=True)

print(df.columns)
df.head()

# =========== Adjust above =============


# %%
# lemmatize the texts using spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
nlp.max_length = 10945772 
spacy.require_cpu()

def lemmatize_docs(texts):
    lemmas_list = []
    for doc in tqdm(nlp.pipe(texts, batch_size=800), total=len(texts)):
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        lemmas_list.append(lemmas)
    return lemmas_list

# %%

# apply lemmatization
lemmatized = lemmatize_docs(df['text'].tolist())

# add lemmatized texts to df
df['lemmatized_text'] = lemmatized

# okay, let's drop the original text column to save space
df = df.drop(columns=['text'])

# save the cleaned metadata with texts and lemmatizations
df.to_json(OUT_DIR / SAVENAME, orient='records', force_ascii=False)



# %%
