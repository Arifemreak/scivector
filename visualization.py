import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

# ✅ Her domain için ortalama embedding vektörü hesapla
def compute_domain_centroids(df, embeddings, domain_col="Domain_Final"):
    domain_vectors = {}
    for domain in df[domain_col].unique():
        idxs = df[df[domain_col] == domain].index
        vecs = [embeddings[i] for i in idxs]
        domain_vectors[domain] = np.mean(vecs, axis=0)
    return domain_vectors

# ✅ t-SNE Görselleştirme
def plot_centroid_tsne(domain_vectors):
    labels = list(domain_vectors.keys())
    vecs = np.array([domain_vectors[d] for d in labels])
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    vecs_2d = tsne.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("tab10", len(labels))

    for i, label in enumerate(labels):
        x, y = vecs_2d[i]
        ax.scatter(x, y, s=80, color=colors[i])
        ax.text(x + 1, y + 1, label, fontsize=10)

    ax.set_title("t-SNE Projection of Domain Centroids")
    ax.grid(True)
    st.pyplot(fig)

# ✅ Cosine Similarity Heatmap
def plot_centroid_similarity_heatmap(domain_vectors):
    labels = list(domain_vectors.keys())
    matrix = cosine_similarity([domain_vectors[d] for d in labels])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", ax=ax)
    ax.set_title("Cosine Similarity Between Domain Centroids")
    st.pyplot(fig)

# ✅ Hiyerarşik Kümeleme (Dendrogram)
def plot_dendrogram(domain_vectors):
    labels = list(domain_vectors.keys())
    vecs = [domain_vectors[d] for d in labels]
    linked = linkage(vecs, method="ward")

    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linked, labels=labels, leaf_rotation=45, ax=ax)
    ax.set_title("Hierarchical Clustering of Scientific Domains")
    st.pyplot(fig)
