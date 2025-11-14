# %%
import pandas as pd

# %%
# get our data
norms = pd.read_csv("cleaned_sensorimotor_norms.csv")
path = "../data/fanfics_metadata_with_texts.json"
data = pd.read_json(path)
data.head()

# %%
def get_dict_scores(texts, dictionary, score_key):

    print(f'Loaded lexicon for scoring, len of lexicon:', len(dictionary))

    total_scores, normalized_scores, avg_matched_scores = [], [], []

    for text in texts: # loop over texts
        valid_scores = []

        for token in text: # loop over tokens in text
            if token in dictionary:
                val = dictionary[token].get(score_key) # get the score
                if val is not None:
                    valid_scores.append(val)

        total_score_text = sum(valid_scores)
        normalized_score_text = total_score_text / len(text) if text else 0
        avg_matched_score_text = total_score_text / len(valid_scores) if valid_scores else 0

        total_scores.append(total_score_text)
        normalized_scores.append(normalized_score_text)
        avg_matched_scores.append(avg_matched_score_text)

    return total_scores, normalized_scores, avg_matched_scores

# %%

# now, for lemma in data['lemmatized_text'], get the sensorimotor norms
norms_cols = ['auditory.mean', 'gustatory.mean', 'haptic.mean',
       'interoceptive.mean', 'olfactory.mean', 'visual.mean', 'foot_leg.mean',
       'hand_arm.mean', 'head.mean', 'mouth.mean', 'torso.mean']

norms_lookup = norms.set_index('word')[norms_cols].to_dict(orient='index')

# %%
for col in norms_cols:
    print(f'Processing scores for modality: {col}')
    total_scores, normalized_scores, avg_matched_scores = get_dict_scores(
        data['lemmatized_text'],
        norms_lookup,
        col
    )
    data[f'total_{col}'] = total_scores
    data[f'normalized_{col}'] = normalized_scores
    data[f'avg_matched_{col}'] = avg_matched_scores

data.head()

# %%
# save the data with scores
data.to_json('../data/fanfics_metadata_with_sensorimotor_scores.json', orient='records', lines=True)
# %%