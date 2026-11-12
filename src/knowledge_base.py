"""
RAG Knowledge Base Loader
Loads JSON knowledge base files and indexes them into ChromaDB vector store.
"""

import os
import json
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class KnowledgeBaseLoader:
    """Loads and indexes college admission documents into ChromaDB."""

    def __init__(self, kb_path: str, vector_db_path: str):
        self.kb_path = Path(kb_path)
        self.vector_db_path = vector_db_path

        # Local sentence-transformers embedding (free, no API cost)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.client.get_or_create_collection(
            name="college_admission",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def load_all(self) -> int:
        """Load all JSON files from the knowledge base directory. Returns doc count."""
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base path not found: {self.kb_path}")

        json_files = list(self.kb_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {self.kb_path}")

        total = 0
        for json_file in json_files:
            count = self._load_file(json_file)
            total += count
            logger.info(f"Loaded {count} documents from {json_file.name}")

        logger.info(f"Total documents indexed: {total}")
        return total

    def _load_file(self, filepath: Path) -> int:
        """Parse a knowledge base JSON file and upsert into ChromaDB."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = data.get("documents", [])
        if not documents:
            return 0

        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            ids.append(doc["id"])
            # Combine title + content for richer embedding
            texts.append(f"{doc['title']}\n\n{doc['content']}")
            metadatas.append({
                "id": doc["id"],
                "category": doc.get("category", "general"),
                "title": doc["title"],
                "source_file": filepath.name,
            })

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
        return len(ids)

    def is_indexed(self) -> bool:
        """Check if the vector store already has documents."""
        return self.collection.count() > 0

    def get_collection(self):
        return self.collection

    def count(self) -> int:
        return self.collection.count()
