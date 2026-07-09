# %%
import pandas as pd
from pathlib import Path

# %%

CWD = Path(__file__).parent
DATA_PATH = CWD.parent / "data"
RESOURCES_PATH = CWD.parent / "resources"

OUTPUT_PATH = DATA_PATH / "scored_data"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %%

# get our data
norms = pd.read_csv(RESOURCES_PATH / "cleaned_sensorimotor_norms.csv")
path = DATA_PATH / "simplestories_lemmatized.json"#"../data/fanfics_metadata_with_texts.json"
data = pd.read_json(path)
data.head()


# %%
def get_dict_scores(texts, dictionary, score_key):

    print(f'Loaded lexicon for scoring, len of lexicon:', len(dictionary))

    total_scores, normalized_scores, avg_matched_scores = [], [], []
    avg_matched_scores_sds = []

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
        avg_matched_score_sd = pd.Series(valid_scores).std() if valid_scores else 0


        total_scores.append(total_score_text)
        normalized_scores.append(normalized_score_text)
        avg_matched_scores.append(avg_matched_score_text)
        avg_matched_scores_sds.append(avg_matched_score_sd)

    return total_scores, normalized_scores, avg_matched_scores, avg_matched_scores_sds

# %%

# now, for lemma in data['lemmatized_text'], get the sensorimotor norms
norms_cols = ['auditory.mean', 'gustatory.mean', 'haptic.mean',
       'interoceptive.mean', 'olfactory.mean', 'visual.mean', 'foot_leg.mean',
       'hand_arm.mean', 'head.mean', 'mouth.mean', 'torso.mean']

norms_lookup = norms.set_index('word')[norms_cols].to_dict(orient='index')

# %%
for col in norms_cols:
    print(f'Processing scores for modality: {col}')
    total_scores, normalized_scores, avg_matched_scores, avg_matched_scores_sds = get_dict_scores(
        data['lemmatized_text'],
        norms_lookup,
        col
    )
    data[f'total_{col}'] = total_scores
    data[f'normalized_{col}'] = normalized_scores
    data[f'avg_matched_{col}'] = avg_matched_scores
    data[f'avg_matched_{col}_sd'] = avg_matched_scores_sds

data.head()

# %%
savename = OUTPUT_PATH / "simplestories_lemmatized_with_scores.json"
# save the data with scores
data.to_json(savename, orient='records', lines=True)
# %%