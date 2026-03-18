# 🚀 RAG, Embeddings & Vector DB – Interview Preparation (FAANG Level)

---

## 🧠 1. Core System Overview

### Q: Explain the end-to-end pipeline of your system.

**Answer:**
I built a document retrieval system where a large PDF is processed through the following steps:

- Extract text from PDF
- Chunk the text into smaller pieces
- Convert each chunk into embeddings
- Store embeddings in a vector database
- At query time, convert the query into an embedding
- Retrieve similar chunks using cosine similarity

---

## 🧠 2. Embeddings

### Q: What are embeddings?

**Answer:**
Embeddings are numerical vector representations of text that capture semantic meaning.

---

## 🧠 3. Cosine Similarity

### Q: What is cosine similarity?

**Answer:**
It measures the angle between two vectors to determine similarity.

---

## 🧠 4. Chunking Strategy

### Q: What chunking strategy did you use?

**Answer:**
Fixed-size chunking with overlap (sliding window).

---

## 🧠 5. Vector Database

### Q: What is a vector database?

**Answer:**
A database optimized for storing embeddings and performing similarity search.

---

## 🧠 6. Persistence Issue

### Q: What issue did you face?

Using the wrong client prevented data from being persisted.

### Fix:
Use PersistentClient.

---

## 🧠 7. Large Scale Processing

- Handled ~10k chunks
- Used batching for embedding and insertion

---

## 🧠 8. Query System

- Convert query → embedding
- Retrieve top-k similar chunks

---

## 🧠 9. Distance vs Similarity

Similarity = 1 - distance

---

## 🧠 10. Real Challenges

- Persistence bugs
- Large data handling
- Performance bottlenecks

---

## 🧠 11. System Design

Separated ingestion and query pipelines.

---

## 🧠 12. Final Statement

I built a scalable document retrieval system using embeddings and vector databases to enable semantic search over large documents.

---

## 🔥 Key Takeaways

- Embeddings = meaning
- Cosine similarity = comparison
- Chunking = precision
- Vector DB = retrieval
- RAG = next step
