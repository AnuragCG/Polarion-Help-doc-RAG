# 💬 Interview Q&A (Deep Dive)

## 🔹 What is RAG?

**Answer:**
Retrieval-Augmented Generation (RAG) is a system design pattern where:

1. Relevant documents are retrieved from a knowledge base
2. The retrieved context is passed to an LLM
3. The LLM generates an answer grounded in that context

It helps reduce hallucinations and enables domain-specific question answering.

---

## 🔹 How does your RAG pipeline work?

**Answer:**
My pipeline consists of the following steps:

1. User query is embedded using a sentence transformer
2. Hybrid retrieval is performed:

   * Dense search (semantic)
   * BM25 (keyword-based)
3. Results are merged and deduplicated
4. Cross-encoder reranker improves precision
5. Context compression extracts relevant sentences
6. LLM generates an answer using strict grounding instructions
7. Evaluation is performed offline

---

## 🔹 Why did you move from basic RAG to advanced pipeline?

**Answer:**
The basic pipeline (dense retrieval + LLM) had limitations:

* Retrieved irrelevant chunks
* Missed keyword-heavy queries
* No way to measure performance

To address this, I incrementally added:

* Reranking → improve precision
* Hybrid search → improve recall
* Compression → reduce noise
* Evaluation → measure improvements

---

## 🔹 What problem does Hybrid Search solve?

**Answer:**
Dense embeddings are good for semantic understanding but weak for exact matches like:

* Error codes
* File names
* Configuration keys

BM25 handles exact keyword matching.

Combining both ensures:

* Semantic coverage
* Lexical precision

---

## 🔹 Why is reranking necessary?

**Answer:**
Initial retrieval focuses on recall and may include loosely related chunks.

Reranking uses a cross-encoder to:

* Evaluate query-document pairs jointly
* Improve precision
* Select the most relevant chunks

---

## 🔹 What is context compression and why is it needed?

**Answer:**
Retrieved chunks often contain noise.

Compression:

* Extracts only relevant sentences
* Reduces token usage
* Improves answer clarity

Without compression, the LLM may get distracted by irrelevant information.

---

## 🔹 What evaluation metrics did you use?

**Answer:**
I implemented:

1. **Recall@K**

   * Checks if expected page is retrieved

2. **Relaxed Recall**

   * Checks semantic correctness using content

3. **Keyword Score**

   * Simple answer quality proxy

4. **LLM-based Faithfulness**

   * Verifies if answer is grounded in context

---

## 🔹 Why is Recall@K not enough?

**Answer:**
Because multiple sections in a document can contain valid answers.

Strict page matching may show low recall even when:

* Correct information is retrieved
* Answer is accurate

That’s why I added relaxed recall and LLM-based evaluation.

---

## 🔹 What is LLM-based evaluation?

**Answer:**
It uses another LLM to verify whether the generated answer is supported by the retrieved context.

This helps detect hallucinations and provides a more reliable correctness signal than keyword matching.

---

## 🔹 How do you debug RAG failures?

**Answer:**
I follow a structured approach:

1. **Retrieval failure**

   * Correct chunk not retrieved

2. **Ranking failure**

   * Correct chunk retrieved but ranked low

3. **Generation failure**

   * LLM misinterprets context

This helps isolate where the pipeline is breaking.

---

## 🔹 What tradeoffs exist in your system?

| Component      | Tradeoff                         |
| -------------- | -------------------------------- |
| Higher Top-K   | Better recall, slower            |
| Reranker       | Better precision, higher latency |
| Hybrid search  | Better coverage, more complexity |
| Compression    | Lower tokens, risk losing info   |
| LLM evaluation | Better accuracy, higher cost     |

---

## 🔹 What were the biggest challenges?

**Answer:**

* Multiple valid answer locations made evaluation difficult
* Dense search missed keyword-heavy queries
* Balancing recall vs precision
* Designing meaningful evaluation metrics

---

## 🔹 What improvements gave the biggest gains?

**Answer:**

* Reranking significantly improved precision
* Hybrid search improved recall for keyword queries
* Evaluation helped identify real bottlenecks

---

## 🔹 What would you improve next?

**Answer:**

* Query rewriting using LLM
* Larger evaluation dataset
* Better chunking strategies (overlap, structure-aware)
* LLM-based answer validation in production
* Latency and cost optimization

---

## 🔹 How would you make this production-ready?

**Answer:**

* Replace BM25 with scalable search (Elasticsearch/OpenSearch)
* Add caching for embeddings and responses
* Add monitoring and logging
* Optimize latency (batching, async calls)
* Add fallback mechanisms

---

## 🔹 What did you learn from this project?

**Answer:**

* Retrieval quality is the most critical component in RAG
* Evaluation is essential to measure improvements
* Simple pipelines fail in real-world scenarios
* Debugging RAG requires separating retrieval and generation

---

## 🔹 Final summary (Strong closing answer)

**Answer:**

> I started with a basic RAG pipeline using dense retrieval and gradually improved it by adding hybrid search, reranking, context compression, and evaluation metrics.
> I focused on measuring performance using Recall@K and LLM-based faithfulness, which helped me understand and fix retrieval and generation issues.
> This project helped me build a strong understanding of how real-world RAG systems are designed and evaluated.
