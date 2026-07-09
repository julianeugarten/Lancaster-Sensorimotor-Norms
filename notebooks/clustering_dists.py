
# %%
import pandas as pd
import pathlib
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN

# import kl divergence
from scipy.special import kl_div

# %%

CWD = pathlib.Path.cwd().parent
RESOURCES_DIR = CWD / "resources"
DATA_DIR = CWD / "data"
FIGS = CWD / "figs"
FIGS.mkdir(exist_ok=True)

senses = ['auditory', 'interoceptive', 'gustatory', 'olfactory', 'haptic', 'visual']

df = pd.read_csv(DATA_DIR / "2025-06-09_combined_sense_profiles.csv")
print(df.shape)


fanfic = df[df['source'] == 'fanfic']
chic = df[df['source'] == 'chicago']

# Empirical shape space
for data in [fanfic, chic]:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data[senses], fill=True)
    plt.title(f"Density of Sense Profiles for {data['source'].iloc[0].capitalize()}")
    plt.xlabel("Sense")
    plt.ylabel("Density")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


df.head()


# %%

# add a vector of residuals

def get_representation(df, senses):
    X = df[senses].values
    X = X / X.sum(axis=1, keepdims=True)

    expected = X.mean(axis=0)

    residuals = X - expected
    residuals_scaled = StandardScaler().fit_transform(residuals)

    return {
        "compositional": X,
        "residuals": residuals,
        "residuals_scaled": residuals_scaled,
        "expected": expected
    }


sample = df.sample(10000, random_state=42) # take a random sample of 1000 books to make it more manageable
representations = get_representation(sample, senses)
representations.keys()


# %%

X = representations["compositional"] # choose which representation to use for clustering

def get_js_distance(X):
    kl_divergence = kl_div(X[:, np.newaxis, :], X[np.newaxis, :, :]).sum(axis=2) # compute KL divergence between all pairs of distributions
    js_distance = np.sqrt(0.5 * kl_divergence + 0.5 * kl_div(X[np.newaxis, :, :], X[:, np.newaxis, :]).sum(axis=2)) # compute JS distance between all pairs of distributions
    return js_distance

dist_matrix = get_js_distance(X)
#dist_matrix = pairwise_distances(X, metric='euclidean')
#dist_matrix = pairwise_distances(X, metric='cosine')

def cluster(dist_matrix, method="hdbscan"):
    if method == "hdbscan":
        model = HDBSCAN(metric="precomputed", min_cluster_size=10)
    else:
        model = AgglomerativeClustering(
            n_clusters=3,
            linkage="average",
            metric="precomputed")
    return model.fit_predict(dist_matrix)

labels = cluster(dist_matrix, method="agglomerative")

sample['cluster'] = labels
print(sample['cluster'].value_counts())

# %%

# see the archetypes of the clusters by looking at the mean profile of each cluster
cluster_means = representations["compositional"].copy()
cluster_means = pd.DataFrame(cluster_means, columns=senses)
cluster_means['cluster'] = labels
cluster_means = cluster_means.groupby('cluster').mean()

global_mean = X.mean(axis=0)

# normalize the cluster means to sum to 1
print(cluster_means)

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

for cluster in cluster_means.index:
    plt.plot(senses, cluster_means.loc[cluster], label=f'Cluster {cluster}')

# also plot the background mean profile
if CONFIG == "residuals":
    base = expected - expected.mean() # center the expected profile to match the residuals
elif CONFIG == "residuals_scaled":
    base = StandardScaler().fit_transform(expected.reshape(1, -1)).flatten() 
else:
    base = sample[senses].mean().values

plt.plot(senses, base, label='Background Mean', linestyle='--', color='grey')

plt.xlabel("Sense")
plt.ylabel("Mean Normalized Rating")
plt.legend()
plt.tight_layout()
plt.show()


# %%


# %%
