---

# 🕉️ GitaInsight AI

### 🛠️ Software & Tools Requirements

1. [VS Code](https://code.visualstudio.com)
2. [Python 3.10+](https://www.python.org)
3. [Google Generative AI SDK](https://www.google.com/search?q=https://github.com/google/generative-ai-python)
4. [ChromaDB](https://www.trychroma.com/)

---

## 📌 Overview

**GitaInsight AI** is an intelligent spiritual companion that leverages **Retrieval-Augmented Generation (RAG)** to provide empathetic, context-aware guidance based on the *Bhagavad-gita As It Is*.

The system features a **hybrid retrieval engine** combining semantic vector search (ChromaDB) with keyword-based sparse search (BM25), fused via **Reciprocal Rank Fusion (RRF)** to ensure high-precision verse retrieval, paired with **Gemma-4 (31B)** for nuanced, therapist-like summarization.

---

## 🧠 Core Architecture

* **Data Pipeline**: Automated PDF ingestion, IAST/Devanagari transliteration, and JSON structuring.
* **Hybrid Retrieval**: Dual-engine search (ChromaDB + BM25) for comprehensive content discovery.
* **Inference**: RRF-based rank fusion for relevance, powered by Google's Gemma-4 LLM.
* **Interface**: Modern Flask-based web application with immersive "floating" glassmorphism UI.

---

## 🗂️ Project Structure

```text
GitaInsight AI/
├── artifacts/             # Not shown, but contains logs/checkpoints
├── chroma_db/             # Persistent vector store
├── data/
│   ├── Bhagavad-gita-As-It-Is.pdf
│   └── gita_structured.json
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # PDF Parser & JSON Generator
│   │   ├── vectorization_and_chunking.py # ChromaDB & BM25 engine
│   │   └── hybrid_search_engine.py  # RRF Logic & LLM query rewriting
│   ├── exception.py                 # Custom error handling
│   └── logger.py                    # Structured logging
├── app.py                           # Flask Web Server
├── index.html                       # Frontend (Floating Glassmorphism UI)
├── cli_tester.py                    # Backend validation pipeline
└── .env                             # API Credentials

```

---

## ⚙️ Tech Stack

| Category | Tools |
| --- | --- |
| **AI / LLM** | Google GenAI SDK (Gemma-4-31B, Gemini-2.5-Flash) |
| **Vector DB** | ChromaDB, Sentence-Transformers |
| **Search Engine** | BM25 (Rank-BM25), RRF Math |
| **Web Framework** | Flask |
| **Data Tools** | PDFPlumber, Indic-Transliteration |

---

## 🚀 Pipeline Workflow

### 1. Data Preparation

* **Ingestion**: Parses `Bhagavad-gita-As-It-Is.pdf` using `GitaIngestor`.
* **Normalization**: Automatically maps legacy BBT fonts to standard IAST Unicode and generates Devanagari script using `indic-transliteration`.

### 2. Hybrid Retrieval Engine

* **Semantic Path**: Chunks verses and stores them in ChromaDB embeddings.
* **Sparse Path**: Indexes verses via BM25 for keyword precision.
* **RRF Fusion**: Combines search results to ensure the most philosophically relevant verse appears at the top.

### 3. Intelligence Layer

* **Query Rewriting**: Uses Gemma-4 to expand user queries into philosophical concepts (`Dharma`, `Karma`, `Equanimity`).
* **Guidance Generation**: Synthesizes retrieved shlokas into empathetic, personalized spiritual advice.

---

## 🧪 How to Run

### 1. Environment Setup

```bash
python -m venv .venv
# Activate virtual environment
pip install -r requirements.txt

```

### 2. Backend Validation

Verify your database and search pipelines are functional:

```bash
python cli_tester.py

```

### 3. Launch Web Interface

```bash
python app.py

```

*Access the interface at: **[http://127.0.0.1:5000*](http://127.0.0.1:5000)**

---

## ⚠️ Common Troubleshooting

| Issue | Solution |
| --- | --- |
| `UnicodeEncodeError` | This is a Windows console emoji rendering issue. It does not affect functionality. |
| `500 Internal Error` | Google's free-tier LLM API is experiencing high traffic. The system has built-in retries. |
| UI is blank | Ensure `index.html` is in the same directory as `app.py`. |

---

## 👨‍💻 Author

**NAYAN BADGUJAR**

---

## ⭐ Acknowledgements

* [Google Generative AI](https://aistudio.google.com/)
* [ChromaDB](https://www.trychroma.com/)
* [Flask](https://flask.palletsprojects.com/)
