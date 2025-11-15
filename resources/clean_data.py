# %%
import pandas as pd
import spacy
from tqdm import tqdm
# %%
## 1st part: process sensorimotor norms ##

# sensorimotor norms downloaded from https://osf.io/rwhs6/overview
norms = pd.read_csv("Lancaster_sensorimotor_norms_for_39707_words.csv")
norms.columns = [col.lower() for col in norms.columns] # lowercase lemmas
norms['word'] = norms['word'].str.lower() # lowercase words
# get only what we need
norms = norms[['word', 'auditory.mean', 'gustatory.mean', 'haptic.mean',
       'interoceptive.mean', 'olfactory.mean', 'visual.mean', 'foot_leg.mean',
       'hand_arm.mean', 'head.mean', 'mouth.mean', 'torso.mean', 'auditory.sd',
       'gustatory.sd', 'haptic.sd', 'interoceptive.sd', 'olfactory.sd',
       'visual.sd', 'foot_leg.sd', 'hand_arm.sd', 'head.sd', 'mouth.sd',
       'torso.sd']].copy()
norms.head(50)

# export to csv
norms.to_csv("cleaned_sensorimotor_norms.csv", index=False)
# %%

## 2nd part: process fanfic corpus ##

# clean corpus
meta = pd.read_csv('../data/fanfics_Greek_myth_metadata.csv')
print(f'Initial metadata has {len(meta)} entries.')
# make ids strings
meta['work_id'] = meta['work_id'].astype(str)
# print length
print("org data has: ", len(meta))

# check which ids do not exist as text files
for id in meta['work_id']:
    if not os.path.exists(f'../data/MythFic_txt/{id}.txt'):
        print(f'Missing file for id: {id}')
# drop rows with missing text files
meta = meta[meta['work_id'] != "38183230"]
print(f'Cleaned metadata has {len(meta)} entries.')

# %%
# texts are in mythfict_txt folder, txt files named by their 'id' in metadata
import os
texts = []
for fid in tqdm(meta['work_id']):
    with open(os.path.join('../data/MythFic_txt', f'{fid}.txt'), 'r', encoding='utf-8') as f:
        text = f.read()
        texts.append(text)
meta['text'] = texts
meta.head()

# %%
# let's lemmatize the texts using spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.max_length = 10945772 
spacy.require_cpu()

def lemmatize_docs(texts):
    lemmas_list = []
    for doc in tqdm(nlp.pipe(texts, batch_size=20), total=len(texts)):
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        lemmas_list.append(lemmas)
    return lemmas_list

# apply lemmatization
lemmatized = lemmatize_docs(meta['text'])

# add lemmatized texts to metadata
meta['lemmatized_text'] = lemmatized
meta.head()

# %%
# save the cleaned metadata with texts and lemmatizations
meta.to_json(
    '../data/fanfics_metadata_with_texts.json',
    orient='records',
    force_ascii=False
)
# %%
