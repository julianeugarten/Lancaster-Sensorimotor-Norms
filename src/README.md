The SRC folder contains the scripts used to process data and generate clean, scored datasets. 

The main scripts are:
- `handle_fanfic_data.py`: Processes fanfics: merges metadata and text data, handles engagement metrics, and generates residuals for age-adjusted analysis.
- `lemmatize.py`: Lemmatizes the text data using the SpaCy model `en_core_web_sm` and generates a lemmatized dataset (in data/lemmatized_data/).
- `modality_score_data.py`: Processes the lemmatized data to calculate modality scores using the sensorimotor norms lexicon, and generates a scored dataset with sensori metrics (in data/scored_data/).