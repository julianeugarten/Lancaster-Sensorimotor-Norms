# Sensorimotor Norms Columns

This dataset contains fanfic-level scores computed using a **sensorimotor norms lexicon**. Each fanfic was tokenized and lemmatized, then matched against the lexicon to compute three types of scores per modality.

## Score Types

For each modality (auditory, gustatory, haptic, interoceptive, olfactory, visual, foot_leg, hand_arm, head, mouth, torso) the following columns were added:

| Column prefix | Description |
|---------------|-------------|
| `total_`      | Sum of all sensorimotor values for the text. Reflects overall intensity across all words. |
| `normalized_` | Total score divided by the **number of words in the text**. Accounts for text length. |
| `avg_matched_` | Total score divided by **number of words matched in the lexicon**. Reflects average intensity for words with sensorimotor values, independent of text length. |

## Modalities

| Modality | Description |
|----------|-------------|
| `auditory.mean` | Auditory-related sensations. |
| `gustatory.mean` | Taste-related sensations. |
| `haptic.mean` | Touch/texture-related sensations. |
| `interoceptive.mean` | Internal bodily sensations (hunger, heartbeat, etc.). |
| `olfactory.mean` | Smell-related sensations. |
| `visual.mean` | Vision-related sensations. |
| `foot_leg.mean` | Foot and leg body-part references. |
| `hand_arm.mean` | Hand and arm body-part references. |
| `head.mean` | Head-related body-part references. |
| `mouth.mean` | Mouth-related body-part references. |
| `torso.mean` | Torso-related body-part references. |

## Notes

- Scores are calculated per fanfic using **lemmatized tokens** to match lexicon entries.  
- `normalized_` adjusts for text length; `avg_matched_` adjusts for lexicon coverage.  
- These scores can be used for studying sensorimotor imagery and embodiment in fanfic texts.


# Sensitive Content Columns

This dataset includes per-fanfic predictions of sensitive content using the **CardiffNLP `twitter-roberta-base-sensitive-multilabel` model**. Each fanfic has been split into chunks (maximum ~510 words) to account for the model’s input length limit, then predictions are aggregated.

## New columns

| Column | Description |
|--------|-------------|
| `sensitive_not-sensitive` | Mean predicted probability that the fanfic contains no sensitive content. Values range 0–1. |
| `sensitive_sex` | Mean predicted probability of sexual content. Values range 0–1. |
| `sensitive_spam` | Mean predicted probability of spam-like content. Values range 0–1. |
| `sensitive_conflictual` | Mean predicted probability of conflictual content (arguments, violence, aggressive language). Values range 0–1. |
| `sensitive_profanity` | Mean predicted probability of profanity or strong language. Values range 0–1. |
| `sensitive_selfharm` | Mean predicted probability of self-harm related content. Values range 0–1. |
| `sensitive_drugs` | Mean predicted probability of drug-related content. Values range 0–1. |
| `sensitivity_prop_above_threshold` | Proportion of chunks across all labels where the predicted probability exceeds a threshold (default 0.8). Provides an overall measure of how much of the text contains highly probable sensitive content. Values range 0–1. |

## Notes

- Probabilities are **averaged across chunks** to provide a single per-fic estimate.  
- Chunking ensures that long texts are fully processed despite the model’s token limit (~512 tokens).  
- The `sensitivity_prop_above_threshold` column summarizes the *intensity* of sensitive content relative to the chosen threshold.
