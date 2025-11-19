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
`sensitivity_prop_above_threshold`: 

# 5. Engagement metrics


# okay
so right now it looks like engagement metrics go down with explicit content, maturity rating, and some senses (haptic mainly). But this might just point to that we see a different form of engagement that is not captured by the engagement ratios.
We'd want to know if they are also negatively associated with raw hits/kudos (log-transformed). If so, then it would point to that these stories just get less engagement overall, not just in terms of ratios (in terms of an engagement form).