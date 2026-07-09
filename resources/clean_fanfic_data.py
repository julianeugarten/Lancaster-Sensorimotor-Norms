# %%
import pandas as pd
import spacy
from tqdm import tqdm
# %%

## process fanfic corpus ##

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
meta.to_json('../data/fanfics_metadata_with_texts.json', orient='records', lines=True)
# %%
