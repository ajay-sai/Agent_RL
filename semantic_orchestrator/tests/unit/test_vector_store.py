"""Tests for vector storage."""
import shutil
from pathlib import Path
from semantic_orchestrator.storage.vector_store import VectorStore
from semantic_orchestrator.types import DataRecord, BackendConfig, StorageBackend

TEST_DIR = Path("test_vector_storage_temp")

def setup():
    """Setup test directory."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

def teardown():
    """Cleanup test directory."""
    if TEST_DIR.exists():
        try:
            shutil.rmtree(TEST_DIR)
        except PermissionError:
            pass  # On Windows, may need manual cleanup later

def test_vector_store_initialization():
    """Test VectorStore can be initialized with config."""
    setup()
    try:
        config = BackendConfig(
            backend_type=StorageBackend.VECTOR,
            connection_string=str(TEST_DIR / "test_chroma"),
            collection_name="test_collection"
        )
        store = VectorStore(config)
        assert store.config == config
        store.close()
    finally:
        teardown()

def test_vector_store_add_records():
    """Test adding records to vector store."""
    setup()
    try:
        config = BackendConfig(
            backend_type=StorageBackend.VECTOR,
            connection_string=str(TEST_DIR / "test_add"),
            collection_name="test_add"
        )
        store = VectorStore(config)

        records = [
            DataRecord(
                id="doc1",
                content="Sample document one about machine learning",
                metadata={"source": "test", "category": "AI"}
            ),
            DataRecord(
                id="doc2",
                content="Sample document two about deep learning",
                metadata={"source": "test", "category": "AI"}
            ),
        ]

        store.add(records)
        assert store.count() == 2
        store.clear()
    finally:
        teardown()

def test_vector_store_search_returns_results():
    """Test that search returns ranked results."""
    setup()
    try:
        config = BackendConfig(
            backend_type=StorageBackend.VECTOR,
            connection_string=str(TEST_DIR / "test_search"),
            collection_name="test_search"
        )
        store = VectorStore(config)
        store.clear()

        records = [
            DataRecord(id="1", content="The quick brown fox jumps over the lazy dog", metadata={}),
            DataRecord(id="2", content="Lazy dog sleeps all day long", metadata={}),
            DataRecord(id="3", content="Foxes are wild animals that hunt", metadata={}),
        ]
        store.add(records)

        results = store.search("fox", k=2)
        assert len(results) == 2
        assert all(r.score > 0 for r in results)
        # Fox-related docs should rank higher
        fox_found = any("fox" in r.content.lower() for r in results)
        assert fox_found, "Expected fox-related document in results"

        store.clear()
    finally:
        teardown()

def test_vector_store_empty_search():
    """Test search with no results."""
    setup()
    try:
        config = BackendConfig(
            backend_type=StorageBackend.VECTOR,
            connection_string=str(TEST_DIR / "test_empty"),
            collection_name="test_empty"
        )
        store = VectorStore(config)
        store.clear()

        results = store.search("nonexistent query", k=5)
        assert len(results) == 0

        store.clear()
    finally:
        teardown()
