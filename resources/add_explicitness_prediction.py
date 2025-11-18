# %%
import pandas as pd
from transformers import pipeline
import time
import numpy as np
from tqdm import tqdm

# %%
# timestamp
ts = time.strftime("%Y-%m-%d_%H-%M")
print("Timestamp:", ts)

df = pd.read_json("../data/fanfics_metadata_with_sensorimotor_scores.json", orient='records', lines=True)

drop_cols = ['total_foot_leg.mean','normalized_foot_leg.mean', 'avg_matched_foot_leg.mean','total_hand_arm.mean', 'normalized_hand_arm.mean',
       'avg_matched_hand_arm.mean', 'total_head.mean', 'normalized_head.mean','avg_matched_head.mean', 'total_mouth.mean', 'normalized_mouth.mean',
       'avg_matched_mouth.mean', 'total_torso.mean', 'normalized_torso.mean','avg_matched_torso.mean']
df = df.drop(columns=drop_cols)
df.head()

# %%

texts = df['text'].astype(str).tolist()

pipe = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sensitive-multilabel",
    top_k=None,
    truncation=True)

labels = ['not-sensitive', 'sex', 'spam', 'conflictual','profanity', 'selfharm', 'drugs']
threshold = 0.8
chunk_size = 510

all_preds = []
prop_above_threshold = []

def predict_chunks(chunks):
    outputs = pipe(chunks, truncation=True)
    # list of dicts per chunk
    chunk_dicts = [{item["label"]: item["score"] for item in pred} for pred in outputs]
    return chunk_dicts

for text in tqdm(texts):
    words = text.split()
    
    # ---- get chunks ----
    if len(words) <= chunk_size:
        chunks = [text] # just one if short enough
    else:
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    # ---- get preds ----
    chunk_dicts = predict_chunks(chunks)
    
    # ---- average scores ----
    avg_scores = {label: np.mean([cd.get(label, 0.0) for cd in chunk_dicts]) for label in labels}
    all_preds.append([{"label": label, "score": avg_scores[label]} for label in labels])
    
    # ---- proportion of chunks above threshold (per label), averaged ----
    per_label_props = []
    for label in labels:
        count = sum(1 for cd in chunk_dicts if cd.get(label, 0.0) >= threshold)
        per_label_props.append(count / len(chunk_dicts))
    
    # final proportion = mean over labels
    prop_above_threshold.append(np.mean(per_label_props))

df["sensitivity_prop_above_threshold"] = prop_above_threshold

# %%
all_preds[0]

first_pred = all_preds[0]
type(first_pred)
# make it a dict for easier access
first_pred_dict = {item['label']: item['score'] for item in first_pred}
first_pred_dict

# %%

# extract label scores
dict_preds = {label: [] for label in labels}

for preds in all_preds:
    preds_dict = {item['label']: item['score'] for item in preds}
    for label in labels:
        dict_preds[label].append(preds_dict.get(label, np.nan))

# add to dataframe
for label in labels:
    df[f'sensitive_{label}'] = dict_preds[label]
df.head()

# %%
# save as JSON version
df.to_json(f"../data/{ts}_fanfics_sensitivity_labelled.json", orient='records', lines=True)
df.head()
# %%
