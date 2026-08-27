"""Main RAG pipeline."""
from typing import List, Dict, Any
from pathlib import Path
from src.rag.config import get_config
from src.rag.ingestion import iter_documents, clean_text
from src.rag.chunking import chunk_with_metadata
from src.rag.embeddings import EmbeddingStore
from src.rag.llm import get_llm_provider, build_prompt, SYSTEM_PROMPT


class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(self):
        self.cfg = get_config()
        self.store = EmbeddingStore()
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm_provider()
        return self._llm

    def ingest(self, docs_dir: Path = None) -> int:
        """Ingest documents from directory into vector store."""
        if docs_dir is None:
            docs_dir = Path(self.cfg.get("DOCUMENTS_DIR", "./data/documents"))

        print(f"Ingesting documents from {docs_dir}...")
        total_chunks = 0

        for filename, text in iter_documents(docs_dir):
            text = clean_text(text)
            print(f"  Processing: {filename} ({len(text)} chars)")

            chunks = list(chunk_with_metadata(
                text,
                source=filename,
                chunk_size=self.cfg["CHUNK_SIZE"],
                overlap=self.cfg["CHUNK_OVERLAP"]
            ))

            self.store.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"    Added {len(chunks)} chunks")

        print(f"Done! Total chunks: {total_chunks}")
        return total_chunks

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system."""
        # Retrieve
        contexts = self.store.query(question, top_k=self.cfg["TOP_K"])

        if not contexts:
            return {
                "answer": "I couldn't find any relevant information in your course materials.",
                "sources": []
            }

        # Generate
        prompt = build_prompt(question, contexts)
        answer = self.llm.generate(prompt, SYSTEM_PROMPT)

        # Format sources
        sources = [
            {
                "source": ctx["source"],
                "chunk_id": ctx["chunk_id"],
                "score": round(ctx["score"], 3)
            }
            for ctx in contexts
        ]

        return {
            "answer": answer,
            "sources": sources
        }

    def stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self.store.get_collection_info()

    def clear(self) -> None:
        """Clear all ingested documents."""
        self.store.clear()
        print("Vector store cleared.")