# backend/ingest_policies.py
import os
import json
import uuid
from tqdm import tqdm


# Prevent TensorFlow imports
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Config
POLICY_FOLDER = os.path.join(os.path.dirname(__file__), "policies")
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Initialize embedding model
print("🚀 Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

# Initialize Chroma client
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="hr_policies")

def ingest_jsonl():
    """Read all JSONL files and embed Q&A text."""
    jsonl_files = [f for f in os.listdir(POLICY_FOLDER) if f.lower().endswith(".jsonl")]
    if not jsonl_files:
        print("⚠️ No JSONL files found in policies folder!")
        return

    for fname in tqdm(jsonl_files, desc="📥 Ingesting policy files"):
        path = os.path.join(POLICY_FOLDER, fname)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())

                    # Handle nested messages or flat question/answer
                    q, a = "", ""
                    if "messages" in item:
                        q = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
                        a = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
                    else:
                        q = item.get("question") or item.get("user") or ""
                        a = item.get("answer") or item.get("assistant") or ""

                    if not q and not a:
                        continue

                    combined = f"Q: {q}\nA: {a}"
                    emb = embedder.encode(combined).tolist()

                    collection.add(
                        ids=[str(uuid.uuid4())],
                        documents=[combined],
                        metadatas=[{"source": fname}]
                    )
                except json.JSONDecodeError:
                    continue

    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest_jsonl()
    print("🎉 All policies embedded into ChromaDB successfully!")
