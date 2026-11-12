"""
Agent Initializer
Sets up the knowledge base, vector store, and RAG engine.
"""

import os
import logging
from dotenv import load_dotenv

from src.knowledge_base import KnowledgeBaseLoader
from src.granite_client import GraniteClient
from src.rag_engine import RAGEngine

load_dotenv()
logger = logging.getLogger(__name__)


def create_agent() -> RAGEngine:
    """
    Initialize and return a ready-to-use RAGEngine.
    Loads knowledge base on first run; reuses existing vector store thereafter.
    """
    kb_path = os.getenv("KNOWLEDGE_BASE_PATH", "./data/knowledge_base")
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vectorstore")
    top_k = int(os.getenv("TOP_K_RESULTS", "3"))

    # Load and index knowledge base
    loader = KnowledgeBaseLoader(kb_path=kb_path, vector_db_path=vector_db_path)

    if not loader.is_indexed():
        logger.info("Vector store is empty — indexing knowledge base...")
        count = loader.load_all()
        logger.info(f"Indexed {count} documents into ChromaDB.")
    else:
        logger.info(f"Vector store ready — {loader.count()} documents loaded.")

    # Initialize IBM Granite client
    granite = GraniteClient()

    # Build and return RAG engine
    return RAGEngine(
        collection=loader.get_collection(),
        granite_client=granite,
        top_k=top_k,
    )
