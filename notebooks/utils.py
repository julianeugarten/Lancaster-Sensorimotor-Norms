
# %%
# 
from scipy.stats import skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt

# %%


### Implementation of polarization measures ###


# A nonparametric entropy-based measure (originally for political polarization), Bao & Gill 2024
# https://www.cambridge.org/core/journals/political-science-research-and-methods/article/nonparametric-entropybased-measure-of-mass-political-polarization/65F5C46042B5FD35444B1AF2EA5A7DD9
# "existing statistical measures of polarization (variances, kurtosis) are not suitable for ordinal variables common in survey data because they assume continuous (interval) data. In addition, nearly all published metrics of polarization are borrowed from statistics because there is some overlap of the statistical properties and the concept of polarization"






# (cluster) group mean + euclidean distance approach
# "the literature has pointed out that this group mean approach is limited in its ability to reflect the full measure of polarization" (https://www.jstor.org/stable/41288382?seq=6)





# data outside median interval approach




# dispersion: to what extent they diverse & "far apart"
# e.g. variance (and basis mean calculation behind it) assumes equal spacing between values (describes dispersion in classic sense)
def variance(values):
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)

# bimodality: to what extent they are clustered around two modes (or more)
# e.g. kurtosis; assumes a single mode, so it can be misleading if there are multiple modes; also sensitive to outliers; also assumes continuous data
# (Mosier et al. 1945)[https://journals.sagepub.com/doi/pdf/10.1177/001316444500500305] proposed the bimodality coefficient to measure bimodality
# "The logic behind the bimodality coefficient is that a bimodal distribution will have high skewness, low kurtosis, or both" (Bao & Gill 2024)
# but it's not suitable for more than 2 modes – assumes bimodality and does not handle trimodality well (Downey & Huffman 2001)[https://www.jstor.org/stable/42955737?seq=1]
def bimodality_coefficient(values):
    n = len(values)
    if n < 3:
        return 0  # Not enough data to compute bimodality
    s = skew(values)
    k = kurtosis(values)  # Use Pearson's definition of kurtosis
    return (s**2 + 1) / (k + (3 * (n - 1)**2) / ((n - 2) * (n - 3)))

def entropy(values):
    if np.issubdtype(values.dtype, np.integer):
        counts = np.bincount(values)
    else:
        # Discretize continuous data into 6 bins
        counts, _ = np.histogram(values, bins=len(VALUES), range=(VALUES[0]-0.5, VALUES[-1]+0.5))
    probabilities = counts / len(values)
    probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
    return -np.sum(probabilities * np.log2(probabilities))




# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis

VALUES = np.arange(6)
NUM_SAMPLES = 10000

# Normalize probability weights
def norm_p(p):
    return np.array(p) / sum(p)

dist_configs = [
    ('uniform', lambda: stats.randint(0, 6).rvs(NUM_SAMPLES), 'samples'),
    ('normal', lambda: stats.norm(loc=2.5, scale=1.2), 'dist'),
    ('exponential', lambda: stats.expon(scale=1.5), 'dist'),
    ('nearly-uniform', lambda: np.random.choice(VALUES, size=NUM_SAMPLES, p=norm_p([0.3, 0.3, 0.3, 0.1, 0.3, 0.3])), 'samples'),
    ('unimodal', lambda: np.random.choice(VALUES, size=NUM_SAMPLES, p=norm_p([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])), 'samples'),
    ('bimodal', lambda: np.random.choice(VALUES, size=NUM_SAMPLES, p=norm_p([0.1, 0.35, 0.1, 0.1, 0.35, 0.1])), 'samples'),
    ('trimodal', lambda: np.random.choice(VALUES, size=NUM_SAMPLES, p=norm_p([0.3, 0.1, 0.3, 0.1, 0.1, 0.3])), 'samples'),
    ('quadrimodal', lambda: np.random.choice(VALUES, size=NUM_SAMPLES, p=norm_p([0.1, 0.3, 0.1, 0.3, 0.3, 0.3])), 'samples'),
]

fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.ravel()

metrics_dict = {}

for ax, (name, dist_fn, dtype) in zip(axes, dist_configs):
    if dtype == 'samples':
        data = dist_fn()
        y = np.bincount(data, minlength=6) / NUM_SAMPLES
    else:
        dist = dist_fn()
        data = dist.rvs(size=NUM_SAMPLES)
        y = dist.pdf(VALUES)
        y = y / y.sum()

    # calculate metrics
    tmp_kurt = kurtosis(data, fisher=True)
    tmp_bimodality = bimodality_coefficient(data)
    tmp_variance = variance(data)
    tmp_entropy = entropy(data)

    metrics_dict[name] = {
        'kurtosis': tmp_kurt,
        'bimodality': tmp_bimodality,
        'variance': tmp_variance,
        'entropy': tmp_entropy}

    ax.bar(VALUES, y, alpha=0.7, edgecolor='black')
    ax.set_title(f"$\mathbf{{{name}}}$\nKurtosis: {tmp_kurt:.2f}\nBimodality Coeff: {tmp_bimodality:.2f}\nVariance: {tmp_variance:.2f}\nEntropy: {tmp_entropy:.2f}")
    ax.set_xticks(VALUES)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

metrics_dict
# %%
