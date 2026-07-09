# %%
import pandas as pd
import spacy
from tqdm import tqdm
# %%
## process sensorimotor norms ##

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

# %%
# export to csv
norms.to_csv("cleaned_sensorimotor_norms.csv", index=False)