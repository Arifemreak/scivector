# 🧬 SciVector: Semantic Expertise Explorer

**SciVector** is a modular, high-performance application for semantic analysis and domain-centric visualization of scientific literature. Designed for researchers, data scientists, and open science initiatives, it leverages transformer-based language models and efficient vector search to explore biomedical knowledge at scale.

---

## 🚀 Key Features

- 🔍 **Semantic Concept Search**  
  Input any scientific term (e.g. *cancer metabolism*, *checkpoint blockade*) and retrieve the most relevant abstracts using cosine similarity over MiniLM embeddings.

- ⚡ **FAISS-Powered Vector Search**  
  Real-time nearest neighbor search over 900+ pre-embedded biomedical abstracts with FAISS L2 indexing.

- 📊 **Scientific Expertise Visualization**  
  Gain insights into how domains relate through interactive:
  - t-SNE projection of domain centroids
  - Cosine similarity heatmap matrix
  - Hierarchical clustering dendrogram

- 📥 **Results Export**  
  Download top search results in CSV format for external analysis.

---

## 🧠 Scientific Motivation

SciVector enables a lightweight form of **expertise mapping** and **semantic exploration** without requiring full-scale LLM inference. It is particularly valuable for:

- Exploratory research synthesis
- Comparative analysis of scientific subfields
- Building semantic bridges across biomedical domains
- Rapid prototyping for AI4Science and NLP-based research tools

---

## 🛠️ Tech Stack

| Layer           | Technology Used                          |
|----------------|-------------------------------------------|
| Embedding      | `sentence-transformers` (MiniLM-L6-v2)   |
| Vector Search  | `FAISS` (L2 distance)                     |
| Visualization  | `t-SNE`, `scikit-learn`, `seaborn`, `matplotlib` |
| Interface      | `Streamlit`                               |

---

## 📂 Project Structure

```bash
scivector/
├── processed_abstracts.csv         # Cleaned biomedical abstracts dataset
├── specter_embeddings.npy          # Precomputed embeddings (MiniLM)
├── streamlit_app.py                # Main interactive UI logic
├── visualization.py                # Domain-level plotting and analysis
├── requirements.txt                # All dependencies
```

---

## 🧪 Try It Live

➡️ **Streamlit App:** [https://scivector.streamlit.app](https://scivector.streamlit.app)
📘 Jupyter Notebook
For a fully annotated, step-by-step breakdown of the SciVector pipeline — including abstract cleaning, transformer embeddings, FAISS indexing, and visualization — check out the accompanying Colab notebook:

➡️ 🧠 Open in Google Colab → Scientific Expertise Mapping.ipynb

This notebook mirrors the full workflow powering the Streamlit app and is ideal for educational use, replication, or modification

---

## 💡 Use Cases

- Researcher profiling & AI-assisted literature review
- Internship and fellowship portfolios for computational biology
- Exploring thematic overlaps between research domains
- Teaching conceptual clustering and NLP-driven knowledge mining

---

## ⚙️ Run Locally

```bash
git clone https://github.com/Arifemreak/scivector.git
cd scivector
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📄 License

**MIT License** — free to use, modify, and distribute.

---

## 👤 Author

Created by [@Arifemreak](https://github.com/Arifemreak), a bioinformatics student and NLP developer focused on bridging artificial intelligence with life sciences for open knowledge.

Feel free to fork, contribute, or reach out if you're building AI tools for science.


