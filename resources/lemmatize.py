# %%

from datasets import load_dataset
import spacy
from tqdm import tqdm
import pandas as pd

# %%
# load a file / dataset
ds = load_dataset("SimpleStories/SimpleStories")
df = pd.DataFrame(ds["train"])
# rename
df = df.rename(columns={"story": "text"})
df.head()

# %%

# sample subset of the data for testing
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# %%
# let's lemmatize the texts using spacy
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

SAVENAME = "simplestories"

# apply lemmatization
lemmatized = lemmatize_docs(df['text'].tolist())

# add lemmatized texts to metadata
df['lemmatized_text'] = lemmatized

# save the cleaned metadata with texts and lemmatizations
df.to_json(f'../data/{SAVENAME}_lemmatized.json', orient='records', force_ascii=False)
# %%








# SETUP FOR BIG

from datasets import load_dataset
import spacy
from tqdm import tqdm
import pandas as pd

# Load full dataset (no sampling)
ds = load_dataset("SimpleStories/SimpleStories")
df = pd.DataFrame(ds["train"])
df = df.rename(columns={"story": "text"})

# Setup spaCy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
nlp.max_length = 10945772
spacy.require_cpu()

def lemmatize_docs(texts):
    lemmas_list = []
    for doc in tqdm(nlp.pipe(texts, batch_size=800), total=len(texts)):
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        lemmas_list.append(lemmas)
    return lemmas_list

SAVENAME = "simplestories_batches"
BATCH_SIZE = 10000
output_path = f'../data/{SAVENAME}_lemmatized.jsonl'

# Process in batches, save incrementally as JSON Lines
with open(output_path, 'w', encoding='utf-8') as f:
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]
        lemmatized = lemmatize_docs(batch['text'].tolist())
        batch['lemmatized_text'] = lemmatized
        batch.to_json(f, orient='records', lines=True, force_ascii=False)
        f.flush()  # Force write to disk