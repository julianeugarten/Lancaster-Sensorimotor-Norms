
# %%
# 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm
from scipy.special import entr


VALUES = np.arange(6)

# --------------------------------------------------
# Distribution definitions
# --------------------------------------------------

def norm_p(p):
    p = np.asarray(p, dtype=float)
    return p / p.sum()

# pure ideal types
ARCHETYPES = {
    "skewed":       [1,0.5,0.25,0.125,0.0625,0.03125],
    "consensus":    [0,0,1,0,0,0],
    "polarized":    [1,0,0,0,0,1],
    "bimodal":      [0,1,0,0,1,0],
    "trimodal":     [1,0,1,0,0,1],
    "~uniform":    [1,1,1,0.5,1,1],
    "uniform":      [1,1,1,1,1,1]}

# level of added noise 
e = 0.05

def add_noise(probs, epsilon=0.05):
    probs = np.asarray(probs, dtype=float)
    noise = np.ones_like(probs) / len(probs) # make sure we get a probability distribution back out
    return (1 - epsilon) * probs + epsilon * noise

def add_random_noise(probs, epsilon=0.05):
    probs = np.asarray(probs, dtype=float)
    noise = np.random.random(len(probs))
    noise /= noise.sum()
    return (1 - epsilon) * probs + epsilon * noise

# normalize to sum to 1 and add noise
DISTS = {name: add_random_noise(norm_p(probs), epsilon=e) for name, probs in ARCHETYPES.items()}

# add a normal dist drawn from a normal distribution with mean 3 and std 1, then normalized to sum to 1
x = np.arange(6)
DISTS["normal"] = norm_p(norm.pdf(x, loc=2.5, scale=1))

# --------------------------------------------------
# Metrics computed directly from probabilities
# --------------------------------------------------

# N: mean-based measures don't mean a thing here... actually
# N: not when dealing with survey data

def mean_from_probs(values, probs):
    return np.sum(values * probs)

# dispersion: to what extent they diverse & "far apart"
# e.g. variance (and basis mean calculation behind it) assumes equal spacing between values (describes dispersion in classic sense)
def variance_from_probs(values, probs):
    mu = mean_from_probs(values, probs)
    return np.sum(probs * (values - mu)**2)

def std_from_probs(values, probs):
    return np.sqrt(variance_from_probs(values, probs))

def entropy_from_probs(probs):
    return np.sum(entr(probs)) / np.log(2)

def coherence(probs):
    """Herfindahl concentration index"""
    return np.sum(probs**2)

def effective_categories(probs):
    """Inverse Herfindahl"""
    return 1 / coherence(probs)

# --------------------------------------------------
# Sample-based shape statistics
# --------------------------------------------------

def draw_sample(probs, n=10000):
    return np.random.choice(VALUES, size=n, p=probs)

# bimodality: to what extent they are clustered around two modes (or more)
# e.g. kurtosis; assumes a single mode, so it can be misleading if there are multiple modes; also sensitive to outliers; also assumes continuous data
# (Mosier et al. 1945)[https://journals.sagepub.com/doi/pdf/10.1177/001316444500500305] proposed the bimodality coefficient to measure bimodality
# "The logic behind the bimodality coefficient is that a bimodal distribution will have high skewness, low kurtosis, or both" (Bao & Gill 2024)
# but it's not suitable for more than 2 modes – assumes bimodality and does not handle trimodality well (Downey & Huffman 2001)[https://www.jstor.org/stable/42955737?seq=1]

# bimodality as Monte Carlo estimate 
def bimodality_coefficient(sample):
    n = len(sample)
    g = skew(sample)
    # Pearson kurtosis (not fisher)
    k = kurtosis(sample, fisher=False)
    return (g**2 + 1) / (k + 3*((n-1)**2)/((n-2)*(n-3)))

# --------------------------------------------------
# Compute metrics
# --------------------------------------------------

results = {}

for name, probs in DISTS.items():

    sample = draw_sample(probs)

    results[name] = {
        "variance": variance_from_probs(VALUES, probs),
        #"std": std_from_probs(VALUES, probs),
        "entropy": entropy_from_probs(probs),
        "coherence": coherence(probs),
        "effective_categories": effective_categories(probs),
        "skew": skew(sample),
        "kurtosis": kurtosis(sample, fisher=True),
        "bimodality": bimodality_coefficient(sample),
    }

results


### Implementation of polarization measures ###

# A nonparametric entropy-based measure (originally for political polarization), Bao & Gill 2024
# https://www.cambridge.org/core/journals/political-science-research-and-methods/article/nonparametric-entropybased-measure-of-mass-political-polarization/65F5C46042B5FD35444B1AF2EA5A7DD9
# "existing statistical measures of polarization (variances, kurtosis) are not suitable for ordinal variables common in survey data because they assume continuous (interval) data. In addition, nearly all published metrics of polarization are borrowed from statistics because there is some overlap of the statistical properties and the concept of polarization"

# (cluster) group mean + euclidean distance approach
# "the literature has pointed out that this group mean approach is limited in its ability to reflect the full measure of polarization" (https://www.jstor.org/stable/41288382?seq=6)

# data outside median interval approach


fig, axes = plt.subplots(2, 4, figsize=(14, 6))

for ax, (name, probs) in zip(axes.ravel(), DISTS.items()):
    sns.set_style("whitegrid")
    ax.bar(VALUES, probs)
    ax.set_title(name)
    ax.set_ylim(0, 0.6)
plt.tight_layout()
plt.show()

df = pd.DataFrame(results).T

plt.figure(figsize=(10, 6))
sns.heatmap(
    df[['variance','entropy','coherence','bimodality', 'effective_categories']],
    annot=True,
    cmap='viridis')
plt.title("Metrics computed directly from probabilities")
plt.show()


# %%
