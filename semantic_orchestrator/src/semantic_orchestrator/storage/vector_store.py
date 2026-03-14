"""ChromaDB vector storage backend."""
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from ..types import DataRecord, BackendConfig, StorageBackend, RetrievalResult


class VectorStore:
    """Vector database wrapper using ChromaDB."""

    def __init__(self, config: BackendConfig):
        """
        Initialize vector store.

        Args:
            config: BackendConfig with connection_string (persist directory) and collection_name
        """
        if config.backend_type != StorageBackend.VECTOR:
            raise ValueError(f"Expected VECTOR backend, got {config.backend_type}")

        self.config = config
        persist_dir = Path(config.connection_string)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        collection_name = config.collection_name or "default"
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Lazy-load embedding model
        self._embedder: Optional[SentenceTransformer] = None

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embedder is None:
            model_name = self.config.extra_params.get(
                "embedding_model",
                "all-MiniLM-L6-v2"
            )
            self._embedder = SentenceTransformer(model_name)
        return self._embedder

    def add(self, records: List[DataRecord]) -> None:
        """
        Add records to vector store.

        Args:
            records: List of DataRecord objects
        """
        if not records:
            return

        ids = [r.id for r in records]
        documents = [r.content for r in records]
        # Ensure metadata is non-empty dict (Chroma requirement)
        metadatas = [r.metadata if r.metadata else {"_placeholder": "1"} for r in records]

        # Generate embeddings
        embeddings = self.embedder.encode(documents, convert_to_numpy=True).tolist()

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        k: int = 10,
        where: Optional[dict] = None,
    ) -> List[RetrievalResult]:
        """
        Search for similar documents.

        Args:
            query: Query string
            k: Number of results to return
            where: Optional filter conditions

        Returns:
            List of RetrievalResult objects, sorted by similarity
        """
        # Embed query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True).tolist()

        # Query collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to RetrievalResult objects
        retrieval_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                content = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                # Chroma returns L2 distance; convert to similarity score
                distance = results["distances"][0][i]
                score = 1.0 / (1.0 + distance)

                result = RetrievalResult(
                    document_id=doc_id,
                    content=content,
                    score=score,
                    source_backend=StorageBackend.VECTOR,
                    metadata=metadata,
                )
                retrieval_results.append(result)

        return retrieval_results

    def clear(self) -> None:
        """Delete all documents from collection."""
        # Get all IDs and delete them
        results = self.collection.get(include=[])
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def count(self) -> int:
        """Return number of documents in collection."""
        return self.collection.count()

    def close(self) -> None:
        """Close the ChromaDB client and free resources."""
        if self.client:
            # Chroma doesn't have an explicit close, but we can reset
            self._embedder = None
            self.client = None
