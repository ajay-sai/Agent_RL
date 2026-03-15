"""Tests for retriever agents."""
from pathlib import Path
import tempfile
from semantic_orchestrator.retrieval import (
    VectorRetriever,
    GraphRetriever,
    SQLRetriever,
    create_retriever_for_backend,
)
from semantic_orchestrator.types import BackendConfig, StorageBackend, DataRecord

def test_vector_retriever_creation():
    """Test VectorRetriever can be created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BackendConfig(
            backend_type=StorageBackend.VECTOR,
            connection_string=str(Path(tmpdir) / "chroma"),
            collection_name="test"
        )
        retriever = VectorRetriever(StorageBackend.VECTOR, config)
        assert retriever.backend == StorageBackend.VECTOR
        retriever.close()

def test_sql_retriever_creation():
    """Test SQLRetriever can be created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config = BackendConfig(
            backend_type=StorageBackend.SQL,
            connection_string=str(db_path),
        )
        retriever = SQLRetriever(StorageBackend.SQL, config)
        assert retriever.backend == StorageBackend.SQL
        retriever.close()

def test_retriever_factory():
    """Test create_retriever_for_backend factory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BackendConfig(
            backend_type=StorageBackend.VECTOR,
            connection_string=str(Path(tmpdir) / "chroma"),
            collection_name="test"
        )
        retriever = create_retriever_for_backend(StorageBackend.VECTOR, config)
        assert isinstance(retriever, VectorRetriever)

def test_vector_retriever_end_to_end():
    """Test vector retriever can add and search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BackendConfig(
            backend_type=StorageBackend.VECTOR,
            connection_string=str(Path(tmpdir) / "chroma"),
            collection_name="test_retrieval"
        )
        retriever = VectorRetriever(StorageBackend.VECTOR, config)

        # Create test records
        records = [
            DataRecord(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence",
                metadata={"topic": "ML"}
            ),
            DataRecord(
                id="doc2",
                content="Deep learning uses neural networks with many layers",
                metadata={"topic": "DL"}
            ),
        ]
        # Use underlying store directly to add
        retriever._store.add(records)

        # Search
        results = retriever.search("neural networks", k=1)
        assert len(results) >= 1
        # Should find deep learning doc

        retriever.close()

def test_sql_retriever_end_to_end():
    """Test SQL retriever can add and search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config = BackendConfig(
            backend_type=StorageBackend.SQL,
            connection_string=str(db_path),
        )
        retriever = SQLRetriever(StorageBackend.SQL, config)

        records = [
            DataRecord(
                id="rec1",
                content="Python is a programming language",
                metadata={"type": "language"}
            ),
            DataRecord(
                id="rec2",
                content="Bash is a shell scripting language",
                metadata={"type": "shell"}
            ),
        ]
        retriever._store.add(records)

        results = retriever.search("Python", k=1)
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

        retriever.close()
