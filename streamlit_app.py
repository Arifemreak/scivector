# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import visualization as viz

# ✅ Load dataset and precomputed embeddings
@st.cache_data
def load_data():
    df = pd.read_csv("processed_abstracts.csv")
    embeddings = np.load("specter_embeddings.npy")
    return df, embeddings

# ✅ Build FAISS index for efficient similarity search
@st.cache_resource
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# ✅ Perform semantic search with normalized score threshold
def semantic_search(query, model, index, df, embeddings, top_k=5, min_score=0.0):
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        score = 1 / (1 + dist)
        if score >= min_score:
            results.append({
                "Score": round(score, 4),
                "Domain": df.iloc[idx].get("Domain_Final", "N/A"),
                "Abstract": df.iloc[idx].get("Cleaned_Abstract", "N/A")[:500] + "..."
            })
    return pd.DataFrame(results)

# ✅ Streamlit application interface
def main():
    st.set_page_config(page_title="SciVector: Semantic Explorer", layout="wide")
    st.title("🧬 SciVector: Semantic Expertise Explorer")
    st.markdown("\n**Explore scientific knowledge semantically.**\n\nLeverage precomputed SPECTER embeddings, FAISS similarity search, and domain-level visualizations to navigate expertise across biomedical research domains.")

    # 🔄 Load data and model
    df, embeddings = load_data()
    model = SentenceTransformer("allenai/specter")  # ✅ must match embedding source
    index = build_faiss_index(embeddings)

    # 🔍 Semantic search section
    st.subheader("🔍 Semantic Abstract Search")
    query = st.text_input("Enter a scientific concept:", "cancer metabolism")
    top_k = st.slider("Top-K Results", 1, 20, 5)
    min_score = st.slider("Min Score (0-1)", 0.0, 1.0, 0.0)

    if st.button("Search"):
        results_df = semantic_search(query, model, index, df, embeddings, top_k, min_score)
        st.dataframe(results_df, use_container_width=True)

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv, "semantic_results.csv", "text/csv")

    # 📊 Visualization section
    with st.expander("📊 Visualize Scientific Domain Landscape"):
        domain_vectors = viz.compute_domain_centroids(df, embeddings)

        st.markdown("#### t-SNE Projection of Domain Centroids")
        viz.plot_centroid_tsne(domain_vectors)

        st.markdown("#### Cosine Similarity Heatmap")
        viz.plot_centroid_similarity_heatmap(domain_vectors)

        st.markdown("#### Hierarchical Clustering Dendrogram")
        viz.plot_dendrogram(domain_vectors)

if __name__ == "__main__":
    main()

