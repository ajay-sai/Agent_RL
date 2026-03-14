"""Tests for SQL storage."""
import pytest
from pathlib import Path
import tempfile
from semantic_orchestrator.storage.sql_store import SQLStore
from semantic_orchestrator.types import DataRecord, BackendConfig, StorageBackend

def test_sql_store_initialization():
    """Test SQLStore can be initialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config = BackendConfig(
            backend_type=StorageBackend.SQL,
            connection_string=str(db_path),
        )
        store = SQLStore(config)
        assert store.config == config
        store.close()

def test_sql_store_add_and_search():
    """Test adding records and searching by metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config = BackendConfig(
            backend_type=StorageBackend.SQL,
            connection_string=str(db_path),
        )
        store = SQLStore(config)

        records = [
            DataRecord(
                id="rec1",
                content="Sample text 1",
                metadata={"category": "A", "value": 10}
            ),
            DataRecord(
                id="rec2",
                content="Sample text 2",
                metadata={"category": "B", "value": 20}
            ),
        ]
        store.add(records)

        # Search by category
        results = store.search(query="", where={"category": "A"})
        assert len(results) == 1
        assert results[0].document_id == "rec1"

        store.clear()
        store.close()

def test_sql_store_text_search():
    """Test full-text like search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config = BackendConfig(
            backend_type=StorageBackend.SQL,
            connection_string=str(db_path),
        )
        store = SQLStore(config)

        records = [
            DataRecord(id="1", content="The quick brown fox", metadata={}),
            DataRecord(id="2", content="Lazy dog sleeps", metadata={}),
            DataRecord(id="3", content="Fox jumps over dog", metadata={}),
        ]
        store.add(records)

        # Search by text
        results = store.search(query="fox", k=2)
        assert len(results) >= 1
        assert any("fox" in r.content.lower() for r in results)

        store.clear()
        store.close()

def test_sql_store_count():
    """Test document counting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config = BackendConfig(
            backend_type=StorageBackend.SQL,
            connection_string=str(db_path),
        )
        store = SQLStore(config)
        store.clear()

        records = [
            DataRecord(id=f"rec{i}", content=f"Content {i}", metadata={})
            for i in range(5)
        ]
        store.add(records)
        assert store.count() == 5

        store.clear()
        store.close()
