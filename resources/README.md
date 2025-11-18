# Sensorimotor Norms Columns

This dataset contains fanfic-level scores computed using a **sensorimotor norms lexicon**. Each fanfic was tokenized and lemmatized, then matched against the lexicon to compute three types of scores per modality.

## Score Types

For each modality (auditory, gustatory, haptic, interoceptive, olfactory, visual, foot_leg, hand_arm, head, mouth, torso) the following columns were added:

| Column prefix | Description |
|---------------|-------------|
| `total_`      | Sum of all sensorimotor values for the text. Reflects overall intensity across all words. |
| `normalized_` | Total score divided by the **number of words in the text**. Accounts for text length. |
| `avg_matched_` | Total score divided by **number of words matched in the lexicon**. Reflects average intensity for words with sensorimotor values, independent of text length. |
| `normalized_sense_sum`  | Sum of all normalized sense values |
| `sense_overall_sum`   | Mean of all `total_*` sensorimotor columns |
| `x.mean_percent` | Fraction of the story’s total normalized sensorimotor intensity accounted for by sense X. For example, if `auditory.mean_percent = 0.3`, then 30% of the story’s total normalized sense intensity comes from auditory scores. |
| `sense_entropy` | Entropy of the six normalized sense values (distributional diversity)   |
| `days_since_published`  | Number of days since the work was published and to 2023-01-01 (arbitrary date) |
| `kudos_hits_ratio` | Ratio of kudos to hits (raw conversion rate)      |
| `comment_hits_ratio` | Ratio of comments to hits (raw conversion rate)  |
| `kudos_ratio_resid` | Residual of `kudos_hits_ratio` after regressing out `days_since_published` (age-corrected)   |
| comment_ratio_resid | Residual of `comment_hits_ratio` after regressing out `days_since_published` (age-corrected) |
| `maturity_rating` | Numeric code for maturity rating (0=General, 1=Teen+, 2=Mature, 3=Explicit, NaN=Not Rated) |

## Notes

- Scores are calculated per fanfic using **lemmatized tokens** to match lexicon entries.  
- `normalized_` adjusts for text length; `avg_matched_` adjusts for lexicon coverage.  


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



## Questions
We could use hits and kudos just by taking their residuals after regressing out days since published. But we have to log-transform them first, since they are highly skewed, this would make the regression more reliable (i.e., better linear model).
Would this work?
Barron paper french revoltion do something very similar.
At some point they say the have fitted OLS for each model
