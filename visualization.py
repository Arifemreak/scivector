# visualization.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

def compute_domain_centroids(df, embeddings, domain_col="Domain_Final"):
    domain_vectors = {}
    for domain in df[domain_col].unique():
        idxs = df[df[domain_col] == domain].index
        vecs = [embeddings[i] for i in idxs]
        domain_vectors[domain] = np.mean(vecs, axis=0)
    return domain_vectors

def plot_centroid_tsne(domain_vectors):
    labels = list(domain_vectors.keys())
    vecs = np.array([domain_vectors[d] for d in labels])
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    vecs_2d = tsne.fit_transform(vecs)

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        x, y = vecs_2d[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, label, fontsize=11)
    plt.title("🧭 t-SNE Projection of Domain Centroids")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_centroid_similarity_heatmap(domain_vectors):
    labels = list(domain_vectors.keys())
    matrix = cosine_similarity([domain_vectors[d] for d in labels])
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap='viridis')
    plt.title("🔥 Domain Centroid Similarity Heatmap")
    plt.tight_layout()
    plt.show()

def plot_dendrogram(domain_vectors):
    labels = list(domain_vectors.keys())
    vecs = [domain_vectors[d] for d in labels]
    linked = linkage(vecs, method='ward')
    plt.figure(figsize=(10, 6))
    dendrogram(linked, labels=labels, leaf_rotation=45)
    plt.title("🌿 Domain Conceptual Hierarchy (Dendrogram)")
    plt.tight_layout()
    plt.show()
