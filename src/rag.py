import os
from typing import List, Tuple

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"
TOP_K = 3
OPENAI_MODEL = "gpt-5.2"

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


def retrieve_context(query: str, top_k: int = TOP_K) -> Tuple[str, List[dict]]:
    """Retrieve the most relevant chunks from Chroma."""
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_items: List[dict] = []
    context_parts: List[str] = []

    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else None
        page = metadata.get("page", "N/A")

        retrieved_items.append(
            {
                "page": page,
                "document": doc,
                "distance": distance,
                "similarity": 1 - distance if distance is not None else None,
            }
        )

        context_parts.append(f"[Page {page}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)
    return context, retrieved_items


def generate_answer(query: str, context: str) -> str:
    """Generate answer from retrieved context using OpenAI."""
    system_prompt = (
        "You are a helpful assistant answering questions from a product documentation knowledge base. "
        "Answer only from the provided context. "
        "If the answer is not present in the context, say: "
        "'I could not find that in the provided documentation.' "
        "When useful, mention page numbers from the context."
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
            similarity = item["similarity"]
            similarity_text = f"{similarity:.4f}" if similarity is not None else "N/A"
            print(
                f"{i}. Page {item['page']} | Similarity: {similarity_text}"
            )

        print("\nGenerating answer...\n")
        answer = generate_answer(query=query, context=context)

        print("Answer:")
        print(answer)
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()