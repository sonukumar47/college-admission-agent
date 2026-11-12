"""
RAG Engine
Retrieves relevant context from ChromaDB and generates answers using IBM Granite.
"""

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    id: str
    title: str
    category: str
    content: str
    distance: float


@dataclass
class AgentResponse:
    answer: str
    sources: List[RetrievedDocument]
    query: str


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for college admission queries.
    1. Embed query → retrieve top-K documents from ChromaDB
    2. Build a grounded prompt with context
    3. Call IBM Granite to generate the final answer
    """

    PROMPT_TEMPLATE = """You are a helpful and knowledgeable College Admission Agent for Greenfield University. \
Answer the student's question accurately and concisely using ONLY the provided context below. \
If the answer is not found in the context, say "I don't have that specific information. \
Please contact admissions at admissions@greenfield.edu for more details."

### Context:
{context}

### Student Question:
{question}

### Answer:"""

    def __init__(self, collection, granite_client, top_k: int = 3):
        self.collection = collection
        self.granite = granite_client
        self.top_k = top_k

    def query(self, question: str) -> AgentResponse:
        """Process a student question end-to-end using RAG."""
        # Step 1: Retrieve relevant documents
        retrieved = self._retrieve(question)

        if not retrieved:
            return AgentResponse(
                answer="I couldn't find relevant information. Please contact admissions@greenfield.edu.",
                sources=[],
                query=question,
            )

        # Step 2: Build context block
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            context_parts.append(f"[{i}] {doc.title}\n{doc.content}")
        context = "\n\n".join(context_parts)

        # Step 3: Build prompt and generate answer
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.granite.generate(prompt)

        return AgentResponse(answer=answer, sources=retrieved, query=question)

    def _retrieve(self, query: str) -> List[RetrievedDocument]:
        """Query ChromaDB for the most relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        if not results["ids"] or not results["ids"][0]:
            return docs

        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            full_text = results["documents"][0][i]
            # Extract just the content (after first newline — title\n\ncontent format)
            content = full_text.split("\n\n", 1)[-1] if "\n\n" in full_text else full_text

            docs.append(RetrievedDocument(
                id=doc_id,
                title=meta.get("title", ""),
                category=meta.get("category", ""),
                content=content,
                distance=results["distances"][0][i],
            ))

        return docs
