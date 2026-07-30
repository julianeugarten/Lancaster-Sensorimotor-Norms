# %%
import pandas as pd
from pathlib import Path
# msttr
from lexical_diversity import lex_div as ld

# %%

CWD = Path(__file__).parent
DATA_PATH = CWD.parent / "data" / "lemmatized_data"
RESOURCES_PATH = CWD.parent / "resources"
norms = pd.read_csv(RESOURCES_PATH / "cleaned_sensorimotor_norms.csv")

OUTPUT_PATH = DATA_PATH / "scored_data"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %%
# get our data
filename = "storyscope_lemmatized.json" #"fanfics_mia_lemmatized.json"#simplestories
path = DATA_PATH / filename
data = pd.read_json(path)
print(len(data))
data.head()


# %%
def get_dict_scores(texts, dictionary, score_key):

    print(f'Loaded lexicon for scoring, len of lexicon:', len(dictionary))

    total_scores, normalized_scores, avg_matched_scores = [], [], []
    avg_matched_scores_sds = []
    n_tokens_list, n_valid_scores_list = [], []
    unique_vs_all_lemmas, msttr_lemmas = [], []

    for text in texts: # loop over texts
        valid_scores = []

        n_tokens_list.append(len(text))

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

        n_valid_scores_list.append(len(valid_scores))

        # stylsitics, unique lemmas/all lemmas and msttr lemma
        unique_lemmas = set(text)
        unique_vs_all_lemmas.append(len(unique_lemmas) / len(text) if text else 0)
        msttr = ld.msttr(text, window_length=100) if len(text) >= 100 else None
        msttr_lemmas.append(msttr)

    return total_scores, normalized_scores, avg_matched_scores, avg_matched_scores_sds, n_tokens_list, n_valid_scores_list, unique_vs_all_lemmas, msttr_lemmas

# %%

# now, for lemma in data['lemmatized_text'], get the sensorimotor norms
norms_cols = ['auditory.mean', 'gustatory.mean', 'haptic.mean',
       'interoceptive.mean', 'olfactory.mean', 'visual.mean'] #, 'foot_leg.mean', 'hand_arm.mean', 'head.mean', 'mouth.mean', 'torso.mean']

norms_lookup = norms.set_index('word')[norms_cols].to_dict(orient='index')

# %%

for col in norms_cols:
    print(f'Processing scores for modality: {col}')
    total_scores, normalized_scores, avg_matched_scores, avg_matched_scores_sds, n_tokens, n_valid_scores, unique_vs_all_lemmas, msttr_lemmas = get_dict_scores(
        data['lemmatized_text'],
        norms_lookup,
        col)
    
    data[f'total_{col}'] = total_scores
    data[f'normalized_{col}'] = normalized_scores
    data[f'avg_matched_{col}'] = avg_matched_scores
    data[f'avg_matched_{col}_sd'] = avg_matched_scores_sds

    data['n_tokens'] = n_tokens
    data[f'n_valid_scores_{col}'] = n_valid_scores

    data['unique_vs_all_lemmas'] = unique_vs_all_lemmas
    data['msttr_lemmas'] = msttr_lemmas

    data[f'perc_valid_scores_{col}'] = data[f'n_valid_scores_{col}'] / data['n_tokens']

data.head()

# %%
filename = "testset" + filename
savename = OUTPUT_PATH / filename.replace('.json', '_with_scores.json')
# save the data with scores
data.to_json(savename, orient='records', lines=True)
# %%
data.head()
# %%
