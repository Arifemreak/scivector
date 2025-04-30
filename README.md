# 🧬 SciVector: Semantic Expertise Explorer

SciVector is an interactive semantic search and expertise visualization tool for scientific literature.  
It allows users to explore conceptual relationships between research domains using transformer embeddings, FAISS indexing, and an intuitive Streamlit interface.

## 🚀 Features

- 🔍 **Semantic Abstract Search**  
  Enter a concept (e.g., _cancer metabolism_) and retrieve the most relevant scientific abstracts via vector similarity.

- 🧠 **Fast Vector Indexing with FAISS**  
  Efficient nearest neighbor search on over 900+ SPECTER-based embeddings.

- 📊 **Scientific Landscape Visualizations**  
  - t-SNE projection of domain centroids  
  - Cosine similarity heatmap  
  - Hierarchical clustering dendrogram

- 📥 **Result Export**  
  Download semantic results as CSV.

---

## 🛠️ Tech Stack

- `sentence-transformers` (MiniLM / SPECTER)
- `FAISS` for approximate similarity search
- `scikit-learn`, `seaborn`, `matplotlib` for analysis & plotting
- `Streamlit` for interactive UI

---

## 🧪 Demo

> 🔗 Live app: [scivector.streamlit.app](https://scivector.streamlit.app)

---

## 📂 Files

- `processed_abstracts.csv` — Cleaned biomedical abstracts
- `specter_embeddings.npy` — Precomputed MiniLM/SPECTER embeddings
- `streamlit_app.py` — Frontend logic
- `visualization.py` — Domain-level analysis & plotting functions
- `requirements.txt` — Environment dependencies

---

## 📄 License

MIT — feel free to use, adapt, and contribute.

---

## 👤 Author

Built by [@Arifemreak](https://github.com/Arifemreak) as part of a scientific AI tools initiative for open knowledge and research mapping.
