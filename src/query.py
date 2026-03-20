
import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from memory import SessionMemory
from rag import (
    retrieve_context,
    generate_answer,
    rewrite_query_with_history,
    initialize_retrieval,   # 🔥 IMPORTANT
)

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

# -----------------------------
# OpenAI client (needed for query rewriting)
# -----------------------------
llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Config
# -----------------------------
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"

# -----------------------------
# Load embedding model (optional for debug)
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1",
    trust_remote_code=True
)

# -----------------------------
# Connect to Chroma
# -----------------------------
print("Connecting to Chroma...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

print(f"Loaded collection with {collection.count()} documents")

# -----------------------------
# 🔥 Initialize retrieval (FIXES BM25 ERROR)
# -----------------------------
print("Initializing retrieval components...")
initialize_retrieval()

# -----------------------------
# Memory (session-based)
# -----------------------------
memory = SessionMemory()
session_id = "user_1"

# -----------------------------
# Interactive loop
# -----------------------------
while True:
    query = input("\nAsk a question (type 'exit' to quit): ").strip()

    if query.lower() == "exit":
        print("Exiting...")
        break

    if not query:
        print("Please enter a valid question.")
        continue

    # -----------------------------
    # Get history
    # -----------------------------
    history = memory.get_last_interactions(session_id)

    # -----------------------------
    # Rewrite query using memory
    # -----------------------------
    rewritten_query = rewrite_query_with_history(
        query,
        history,
        llm_client
    )

    print(f"\nRewritten Query: {rewritten_query}")

    # -----------------------------
    # Retrieve context
    # -----------------------------
    context, retrieved_items = retrieve_context(rewritten_query)

    print("\nRetrieved Chunks:")
    for i, item in enumerate(retrieved_items, start=1):
        score = item.get("rerank_score", None)
        score_text = f"{score:.4f}" if score is not None else "N/A"

        print(f"{i}. Page {item['page']} | Score: {score_text}")

    # -----------------------------
    # Generate answer (original query)
    # -----------------------------
    answer = generate_answer(query, context)

    print("\nAnswer:")
    print(answer)

    # -----------------------------
    # Save memory
    # -----------------------------
    memory.add_interaction(session_id, query, answer)
