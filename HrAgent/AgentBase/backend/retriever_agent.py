# backend/retriever_agent.py
import os
from typing import List, Dict
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Prevent TensorFlow / Flax imports (important for Windows)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Config paths
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "hr_policies"

class PolicyRetriever:
    """Retrieves relevant HR policy sections from ChromaDB given a grievance."""

    def __init__(self, db_path: str = CHROMA_DIR, embed_model: str = EMBED_MODEL):
        print("🚀 Loading embedding model for retrieval...")
        self.embedder = SentenceTransformer(embed_model)
        self.client = PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        print("✅ Connected to ChromaDB successfully!")

    def search_policies(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Search the Chroma collection for top_k matching policy entries."""
        query_embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        policies = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            policies.append({
                "policy_text": doc,
                "source": meta.get("source", "unknown"),
                "type": meta.get("type", "")
            })

        return policies


# Run standalone test
if __name__ == "__main__":
    retriever = PolicyRetriever()

    # Example grievance
    grievance_text = "My manager keeps changing my shift timings and being rude to me."

    results = retriever.search_policies(grievance_text, top_k=3)

    print("\n🔍 Retrieved Relevant Policies:")
    for i, p in enumerate(results, 1):
        print(f"\n[{i}] From: {p['source']}")
        print(p['policy_text'][:400], "...\n")
