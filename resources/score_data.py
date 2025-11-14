# %%
import pandas as pd

# %%
# get our data
norms = pd.read_csv("cleaned_sensorimotor_norms.csv")
data = pd.read_csv("../data/cleaned_fanfic_corpus.csv")

# %%
# now, for lemma in data['lemmatized_text'], get the sensorimotor norms
norms_cols = ['auditory.mean', 'gustatory.mean', 'haptic.mean',
       'interoceptive.mean', 'olfactory.mean', 'visual.mean', 'foot_leg.mean',
       'hand_arm.mean', 'head.mean', 'mouth.mean', 'torso.mean']

scores_dict = {col : [] for col in norms_cols}
scores_dict
# %%
tmp = []
for i, row in enumerate(data[:3]):
    text = row['lemmatized_text']
    id = row['work_id']
    for lemma in text:
        if lemma in norms['word'].values:
            score = norms[norms['word'] == lemma][norms_cols]
    # save score
    scores_dict[id] = score
            
# %%



def get_sensorimotor_scores(lemmas, norms):
    # create a dataframe to hold the scores
    scores = pd.DataFrame(columns=[norms_cols])

    # for each lemma, get the scores if it exists in norms
    for lemma in lemmas:
        if lemma in norms['word'].values:
            row = norms[norms['word'] == lemma].iloc[0]
            scores = scores.append(row[1:12], ignore_index=True)
    
    # calculate mean scores for the document
    mean_scores = scores.mean().to_dict()

    return mean_scores

# apply to each document in data
sensorimotor_data = []
for lemmas in data['lemmatized_text']:
    lemmas_list = lemmas.strip("[]").replace("'", "").split(", ")
    scores = get_sensorimotor_scores(lemmas_list, norms)
    sensorimotor_data.append(scores)

sensorimotor_df = pd.DataFrame(sensorimotor_data)