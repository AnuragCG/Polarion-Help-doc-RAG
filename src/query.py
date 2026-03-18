import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"

# -----------------------------
# Load embedding model
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1",
    trust_remote_code=True
)

# -----------------------------
# Load Chroma DB (persistent)
# -----------------------------
print("Connecting to Chroma...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_collection(name=COLLECTION_NAME)

print(f"Loaded collection with {collection.count()} documents")

# -----------------------------
# Interactive query loop
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
    # Embed query
    # -----------------------------
    query_embedding = model.encode([query]).tolist()

    # -----------------------------
    # Retrieve results (with distances)
    # -----------------------------
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # -----------------------------
    # Display results
    # -----------------------------
    print("\nTop results:\n")

    if not documents:
        print("No results found.")
        continue

    for i, doc in enumerate(documents):
        page = metadatas[i].get("page", "N/A") if i < len(metadatas) else "N/A"
        distance = distances[i] if i < len(distances) else None

        # Convert distance → similarity (approx)
        similarity = 1 - distance if distance is not None else None

        print(f"Result {i+1} (Page {page}):")

        if distance is not None:
            print(f"Distance: {distance:.4f} | Similarity: {similarity:.4f}")

        print(doc[:300], "...\n")

    print("-" * 60)