"""Retriever implementations for different storage backends."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..types import (
    DataRecord,
    RetrievalResult,
    StorageBackend,
    BackendConfig,
    QueryPlan,
)
from ..storage import VectorStore, GraphStore, SQLStore


class BaseRetriever(ABC):
    """Abstract base class for retriever agents."""

    def __init__(self, backend: StorageBackend, config: BackendConfig):
        """
        Initialize retriever.

        Args:
            backend: StorageBackend enum value
            config: BackendConfig for connection
        """
        self.backend = backend
        self.config = config
        self._store = self._create_store()

    @abstractmethod
    def _create_store(self):
        """Create the storage backend instance."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Search for relevant documents.

        Args:
            query: Query string
            k: Number of results
            filters: Metadata filters

        Returns:
            List of RetrievalResult
        """
        pass

    def close(self) -> None:
        """Close underlying store connection."""
        if hasattr(self._store, 'close'):
            self._store.close()


class VectorRetriever(BaseRetriever):
    """Retriever for ChromaDB vector store."""

    def _create_store(self) -> VectorStore:
        return VectorStore(self.config)

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        if self._store is None:
            raise RuntimeError("Store not initialized")
        results = self._store.search(query, k=k, where=filters)
        return results


class GraphRetriever(BaseRetriever):
    """Retriever for Neo4j graph store."""

    def _create_store(self) -> GraphStore:
        return GraphStore(self.config)

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        if self._store is None:
            raise RuntimeError("Store not initialized")
        results = self._store.search(query, k=k, where=filters)
        return results


class SQLRetriever(BaseRetriever):
    """Retriever for SQLite store."""

    def _create_store(self) -> SQLStore:
        return SQLStore(self.config)

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        if self._store is None:
            raise RuntimeError("Store not initialized")
        results = self._store.search(query, k=k, where=filters)
        return results


def create_retriever_for_backend(
    backend: StorageBackend,
    config: BackendConfig
) -> BaseRetriever:
    """
    Factory function to create appropriate retriever.

    Args:
        backend: Storage backend type
        config: Backend configuration

    Returns:
        Retriever instance
    """
    if backend == StorageBackend.VECTOR:
        return VectorRetriever(backend, config)
    elif backend == StorageBackend.GRAPH:
        return GraphRetriever(backend, config)
    elif backend == StorageBackend.SQL:
        return SQLRetriever(backend, config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
