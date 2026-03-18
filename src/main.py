import os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

from pdf_loader import load_pdf
from chunker import chunk_text

# -----------------------------
# Config
# -----------------------------
PDF_PATH = os.path.join("data", "Administrator and User Help.pdf")
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"

EMBED_BATCH_SIZE = 32
DB_BATCH_SIZE = 500

# -----------------------------
# Safety check
# -----------------------------
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

# -----------------------------
# Load embedding model
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1",
    trust_remote_code=True
)

# -----------------------------
# Load PDF
# -----------------------------
print("Loading PDF...")
pages = load_pdf(PDF_PATH)

# -----------------------------
# Chunk text
# -----------------------------
print("Chunking text...")
chunks = chunk_text(pages)

documents = [chunk["text"] for chunk in chunks]
metadatas = [{"page": chunk["page"]} for chunk in chunks]
ids = [str(i) for i in range(len(documents))]

print(f"Total chunks: {len(documents)}")

# -----------------------------
# Initialize Chroma (persistent)
# -----------------------------
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(name=COLLECTION_NAME)

print(f"Existing documents in DB: {collection.count()}")

# -----------------------------
# Insert only if empty
# -----------------------------
if collection.count() == 0:

    # --------- EMBEDDING ---------
    print("Creating embeddings...")
    embeddings = []

    for i in tqdm(range(0, len(documents), EMBED_BATCH_SIZE), desc="Embedding"):
        batch = documents[i:i + EMBED_BATCH_SIZE]

        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False
        ).tolist()

        embeddings.extend(batch_embeddings)

    # --------- STORAGE ---------
    print("Storing in Chroma...")
    print("Saving embeddings to disk...")

    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    for i in tqdm(range(0, len(documents), DB_BATCH_SIZE), desc="Inserting"):
        collection.add(
            documents=documents[i:i + DB_BATCH_SIZE],
            embeddings=embeddings[i:i + DB_BATCH_SIZE],
            metadatas=metadatas[i:i + DB_BATCH_SIZE],
            ids=ids[i:i + DB_BATCH_SIZE]
        )

    

    print("✅ Data stored in Chroma!")

else:
    print("⚠️ Data already exists. Skipping embedding.")