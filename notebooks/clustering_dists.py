
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


def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    return np.sum(p * np.log(p / q))

def js_distance(p, q, eps=1e-12):
    p = np.asarray(p)
    q = np.asarray(q)

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    js = 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)

    return np.sqrt(js)

# %%

CWD = pathlib.Path.cwd().parent
RESOURCES_DIR = CWD / "resources"
DATA_DIR = CWD / "data"
FIGS = CWD / "figs"
FIGS.mkdir(exist_ok=True)

senses = ['auditory', 'interoceptive', 'gustatory', 'olfactory', 'haptic', 'visual']

df = pd.read_csv(DATA_DIR / "2025-06-09_combined_sense_profiles.csv")
print(df.shape)
df.head()
# %%


# Empirical shape space

sample = df.sample(10000, random_state=42) # take a random sample of 1000 books to make it more manageable

X = sample[senses].values
X = X + 1e-8 # add small constant to avoid issues with zero probabilities
X = X / X.sum(axis=1, keepdims=True) # normalize to sum to 1

#dist_matrix = pairwise_distances(X, metric='cosine') # or better, J-S divergence, but let's start with cosine distance
dist_matrix = pairwise_distances(X, metric=js_distance)

clustering = AgglomerativeClustering(n_clusters=3, linkage='average', metric='precomputed')
labels = clustering.fit_predict(dist_matrix)

#labels = HDBSCAN(metric='precomputed', min_cluster_size=10).fit_predict(dist_matrix)

sample['cluster'] = labels

print(sample['cluster'].value_counts())



# %%
# PCA

X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

sample['pca1'] = X_pca[:, 0]
sample['pca2'] = X_pca[:, 1]

# project on 2D PCA space and color by cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sample, x='pca1', y='pca2', hue='cluster', palette='Set1')
plt.title("PCA of Sense Profiles Colored by Cluster")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# see the archetypes of the clusters by looking at the mean profile of each cluster
cluster_means = sample.groupby('cluster')[senses].mean()
print(cluster_means)

plt.figure(figsize=(10, 6))
for cluster in cluster_means.index:
    plt.plot(senses, cluster_means.loc[cluster], label=f'Cluster {cluster}')
plt.xlabel("Sense")
plt.ylabel("Mean Normalized Rating")
plt.legend()
plt.tight_layout()
plt.show()
# %%


# GET RESIDS OF distributions (pred by avg dist) to get both general coherence of dist across chic/fanfic
# cluster resids?????