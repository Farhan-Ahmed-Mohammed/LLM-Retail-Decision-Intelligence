import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag_documents.jsonl"
)

CHROMA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chroma_db"
)


# SETTINGS

COLLECTION_NAME = "retail_documents"

# Lightweight embedding model suitable for local development
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# LOAD EMBEDDING MODEL

print("=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)

print(f"Model: {EMBEDDING_MODEL}")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded successfully!")


# CONNECT TO CHROMADB

print("\n" + "=" * 60)
print("CONNECTING TO CHROMADB")
print("=" * 60)

CHROMA_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "Retail RAG document collection"
    }
)

print("ChromaDB connected successfully!")


# LOAD RAG DOCUMENTS

print("\n" + "=" * 60)
print("LOADING RAG DOCUMENTS")
print("=" * 60)

documents = []

with open(DOCUMENT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if line.strip():
            documents.append(json.loads(line))

print(f"Total documents loaded: {len(documents)}")


# PREPARE DATA

ids = []
texts = []
metadatas = []

for document in documents:

    ids.append(document["document_id"])

    texts.append(document["text"])

    metadatas.append({
        "source": str(document["source"]),
        "entity_type": str(document["entity_type"]),
        "product_id": str(document["product_id"])
    })


# CREATE EMBEDDINGS

print("\n" + "=" * 60)
print("CREATING EMBEDDINGS")
print("=" * 60)

print("This may take some time for 103,156 documents...")
print()

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEmbeddings created successfully!")

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")


# STORE DOCUMENTS IN CHROMADB

print("\n" + "=" * 60)
print("STORING DOCUMENTS IN CHROMADB")
print("=" * 60)

# Add documents in batches
# This avoids sending all 103,156 documents in one operation.

BATCH_SIZE = 5000

for start in range(0, len(documents), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(documents))

    print(f"Adding documents {start + 1} to {end}...")

    collection.upsert(
        ids=ids[start:end],
        documents=texts[start:end],
        metadatas=metadatas[start:end],
        embeddings=embeddings[start:end].tolist()
    )


# VERIFY DATABASE

print("\n" + "=" * 60)
print("CHROMADB CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Collection name: {COLLECTION_NAME}")
print(f"Documents in database: {collection.count()}")
print(f"Database location: {CHROMA_DIR}")

print("\n" + "=" * 60)
print("EMBEDDING PIPELINE COMPLETED!")
print("=" * 60)