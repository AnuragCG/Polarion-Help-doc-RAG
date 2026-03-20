import os
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import re
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# -----------------------------
# Config
# -----------------------------
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"
TOP_K = 3
OPENAI_MODEL = "gpt-5.2"
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------
# Load environment
# -----------------------------
load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")
print("Loaded API Key:", api_key)
# -----------------------------
# Clients
# -----------------------------
print("Loading embedding model...")
embedding_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1",
    trust_remote_code=True,
)

print("Connecting to Chroma...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

print(f"Loaded collection with {collection.count()} documents")

print("Connecting to OpenAI...")
llm_client = OpenAI(api_key=api_key)

def rewrite_query_with_history(query, history, llm_client):
    if not history:
        return query

    history_text = "\n".join(
        [f"Q: {h['query']}\nA: {h['answer']}" for h in history]
    )

    prompt = f"""
You are helping improve search queries.

Conversation history:
{history_text}

Current question:
{query}

Rewrite the question so it is self-contained and clear for document search.
Do NOT answer the question.

Return only the rewritten query.
"""

    response = llm_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

def initialize_retrieval():
    global bm25, bm25_docs, bm25_metas
    bm25, bm25_docs, bm25_metas = build_bm25_index()

def compress_documents(query: str, docs: list):
    """Extract most relevant sentences from each document."""
    
    query_terms = set(re.findall(r"\w+", query.lower()))
    compressed_docs = []

    for doc in docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc)

        scored = []
        for sent in sentences:
            sent_terms = set(re.findall(r"\w+", sent.lower()))
            overlap = len(query_terms & sent_terms)
            scored.append((sent, overlap))

        # Sort sentences by relevance
        scored = sorted(scored, key=lambda x: x[1], reverse=True)

        # Keep top sentences (adjustable)
        best_sentences = [s for s, _ in scored[:3]]

        compressed_doc = " ".join(best_sentences)
        compressed_docs.append(compressed_doc)

    return compressed_docs

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def build_bm25_index():
    print("Building BM25 index...")

    results = collection.get(include=["documents", "metadatas"])

    documents = results["documents"]
    metadatas = results["metadatas"]

    tokenized_corpus = [tokenize(doc) for doc in documents]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, documents, metadatas

def retrieve_context(query: str, top_k: int = 10):

    # -----------------------
    # 1. Dense retrieval
    # -----------------------
    query_embedding = embedding_model.encode([query]).tolist()

    dense_results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    dense_docs = dense_results["documents"][0]
    dense_metas = dense_results["metadatas"][0]

    # -----------------------
    # 2. BM25 retrieval
    # -----------------------
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:top_k]

    bm25_docs_selected = [bm25_docs[i] for i in top_indices]
    bm25_metas_selected = [bm25_metas[i] for i in top_indices]

    # -----------------------
    # 3. Combine results
    # -----------------------
    combined_docs = dense_docs + bm25_docs_selected
    combined_metas = dense_metas + bm25_metas_selected

    # Remove duplicates
    unique = {}
    for doc, meta in zip(combined_docs, combined_metas):
        unique[doc] = meta

    final_docs = list(unique.keys())
    final_metas = list(unique.values())

    # -----------------------
    # 4. Rerank
    # -----------------------
    pairs = [(query, doc) for doc in final_docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(final_docs, final_metas, scores),
        key=lambda x: x[2],
        reverse=True,
    )

    top_docs = ranked[:3]
    docs_only = [doc for doc, meta, score in top_docs]

    # 🔥 NEW: compress here
    compressed_docs = compress_documents(query, docs_only)

    # -----------------------
    # 5. Build context
    # -----------------------
    retrieved_items = []
    context_parts = []

    for i, (doc, meta, score) in enumerate(top_docs):
        page = meta.get("page", "N/A") if meta else "N/A"

        retrieved_items.append(
            {
                "page": page,
                "document": doc,
                "rerank_score": float(score),
            }
        )

        compressed_doc = compressed_docs[i]

        context_parts.append(f"[Page {page}]\n{compressed_doc}")

    context = "\n\n---\n\n".join(context_parts)

    return context, retrieved_items

def generate_answer(query: str, context: str) -> str:
    """Generate answer from retrieved context using OpenAI."""
    system_prompt = (
        "You are a strict documentation QA assistant.\n"
        "Answer ONLY using the provided context.\n"
        "Do NOT guess or use outside knowledge.\n"
        "If the answer is not clearly in the context, say exactly:\n"
        "'I could not find that in the provided documentation.'\n"
        "Always include page numbers when possible."
    )

    user_prompt = f"""Question:
{query}

Context:
{context}

Instructions:
- Answer clearly and directly.
- Use only the context above.
- If the context is insufficient, say so.
- Include page references when relevant.
"""

    response = llm_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content or "No answer generated."

def main() -> None:
    initialize_retrieval()

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ").strip()

        if query.lower() == "exit":
            print("Exiting...")
            break

        if not query:
            print("Please enter a valid question.")
            continue

        print("\nRetrieving relevant context...")
        context, retrieved_items = retrieve_context(query=query, top_k=TOP_K)

        print("\nRetrieved Chunks:")
        for i, item in enumerate(retrieved_items, start=1):
            score = item.get("rerank_score", None)
            score_text = f"{score:.4f}" if score is not None else "N/A"

            print(
                f"{i}. Page {item['page']} | Rerank Score: {score_text}"
            )

        print("\nGenerating answer...\n")
        answer = generate_answer(query=query, context=context)

        print("Answer:")
        print(answer)
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()