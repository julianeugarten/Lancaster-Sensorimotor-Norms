# 1. Raw sensorimotor means (normalized_*.mean)
Moderate positive correlations among taste/smell/other senses:
- gustatory ↔ olfactory = 0.84
- auditory ↔ olfactory = 0.51
- haptic ↔ visual = 0.73
These make sense: some sensory modalities often co-occur in descriptions (taste & smell; touch & vision).

*Interoceptive* is positively correlated with auditory, gustatory, and olfactory (0.55–0.61) – all but visual, suggesting works with stronger external senses also often describe internal states. (interesting)
- interoceptive ↔ overall (normalized) sum = 0.7; also suggests that this is the case. The more in overall score a story has, the more interoceptive content it tends to have.
- interoceptive ↔ sense_entropy = 0.71; Suggests that interoceptive content is more common in stories with diverse sensory descriptions. This is also the case for gustatory (0.87) and olfactory (0.77) with sense_entropy.

# 2. Percent columns (*_percent)
These are compositional, so correlations are more complex: increasing one sense automatically decreases the others’ proportion.
Example: auditory_percent vs haptic_percent = -0.82 → strong negative due to normalization.
Many negative correlations among different sense percentages are expected—they reflect trade-offs in relative representation rather than absolute intensity.
*Visual_percent* shows the strongest negative correlations with other senses (e.g., -0.81 with interoceptive_percent), perhaps(?) indicating that when visual descriptions dominate, other sensory modalities are less represented proportionally.
This can also be seen in its strong negative correlation with sense_entropy (-0.84), suggesting that visually-dominant stories tend to have less diverse sensory descriptions overall.

We could look more into this

# 3. Entropy (sense_entropy)
High positive correlations with gustatory.mean, olfactory.mean, and interoceptive.mean (~0.71–0.87)
Interpretation: works that spread attention across multiple senses (especially gustatory/olfactory/interoceptive) have higher entropy.
Slight negative correlation with visual.mean (-0.14) → visual-heavy works are often less multisensory (?), so lower entropy.


# 4. Sensitivity labels
Positive correlations among labels like sex, drugs, selfharm, profanity (~0.19–0.75) indicate that works flagged for one kind of sensitive content often overlap with others.
not-sensitive is negatively correlated with all sensitive labels (-0.55 to -0.70), which is logical, this column indicates absence of sensitive content 0-1 (overall prediction by model).

# 5. Engagement metrics

### What we measure
The dataset includes both raw engagement counts and “conversion-style” ratios:
- Raw metrics: hits, kudos, comments
- Ratios: kudos_hits_ratio — how many readers who clicked also left kudos / comment_hits_ratio — same but for comments
- Residual versions (`x_resid`): ratios and counts after regressing out days_since_published, to correct for platform age effects.

## General patterns
- Hits and kudos are strongly correlated, as expected (.85).
- Ratios behave differently: works with very high traffic tend to accumulate kudos more slowly, so their kudos-per-hit tends to be lower. Kudos are positively correlated with the ratio (.23); hits are negatively correlated with the ratio (-.21): i.e., *we see the difference between popularity (hits) and engagement rate (ratios)*.

## Maturity rating amplifies this:
It correlates positively with hits and kudos.
It correlates negatively with the ratios — meaning higher-rated works attract more eyeballs overall but convert a smaller share of them into kudos or comments.

For more on the maturity ratings relation to engagement, see the histograms in figs folder.

## Sensorimotor Language
Most individual sensory modalities show near-zero association with engagement outcomes. A small exception:
- Haptic has a mild positive association with hits/kudos (resid); it correlates with maturity rating (.32-.36); and with most sensitivity labels (.1-.5)
- Auditory shows similar, but inverse patterns with the same: mild negative association with hits/kudos (resid), maturity rating, and sensitivity labels.

## Sensitivity Labels
The sensitivity classifier (e.g., sex, profanity, drugs, etc.) shows a pattern similar to the maturity rating:
Categories like sex, profanity, and drug-related content show higher hits and kudos,
but lower ratios, suggesting a stronger casual-audience effect (more readers who browse than interact).

In summary:
Maturity rating and sensitivity labels show a consistent pattern: they are positively associated with raw engagement (hits/kudos) but negatively associated with engagement ratios (kudos_hits_ratio/comment_hits_ratio).
