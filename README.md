# 🧠 Polarion RAG System (Retrieval-Augmented Generation)

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline over a large (~3000 pages) Polarion documentation PDF.

The system allows users to:

* Ask natural language questions
* Retrieve relevant documentation chunks
* Generate grounded answers using an LLM

---

## 🚀 Features

* ✅ Dense vector search using embeddings
* ✅ Hybrid retrieval (Dense + BM25)
* ✅ Cross-encoder reranking for precision
* ✅ Context compression for cleaner inputs
* ✅ Grounded answer generation (no hallucinations)
* ✅ Evaluation framework:

  * Recall@K
  * Relaxed Recall
  * Keyword scoring
  * LLM-based faithfulness

---

## 🏗️ Architecture

```
User Query
   ↓
Embedding (Dense Search)
   +
BM25 (Keyword Search)
   ↓
Merge Results
   ↓
Re-ranking (Cross Encoder)
   ↓
Context Compression
   ↓
LLM (Answer Generation)
   ↓
Evaluation (Offline)
```

---

## 📂 Project Structure

```
src/
├── pdf_loader.py       # Load PDF documents
├── chunker.py          # Chunking logic
├── rag.py              # Core RAG pipeline
├── query.py            # CLI interaction
├── evaluator.py        # Evaluation metrics
├── eval_runner.py      # Evaluation runner
├── main.py             # Entry point
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install chromadb sentence-transformers openai python-dotenv rank-bm25
```

---

### 2. Set environment variable

Create `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

### 3. Run the system

```bash
python src/main.py
```

---

### 4. Run evaluation

```bash
python src/eval_runner.py
```

---

## 📊 Evaluation Metrics

| Metric             | Purpose                                |
| ------------------ | -------------------------------------- |
| Recall@K           | Measures if correct page is retrieved  |
| Relaxed Recall     | Checks semantic retrieval correctness  |
| Keyword Score      | Basic answer quality proxy             |
| Faithfulness (LLM) | Verifies answer is grounded in context |

---

## 🧠 Key Learnings

* Retrieval quality is more important than generation
* Hybrid search improves coverage significantly
* Reranking improves precision
* Evaluation is essential for improving RAG systems
* Multiple document sections can contain valid answers

---
