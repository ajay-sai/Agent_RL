"""Tests for graph storage (Neo4j)."""
import pytest
from semantic_orchestrator.storage.graph_store import GraphStore
from semantic_orchestrator.types import DataRecord, BackendConfig, StorageBackend

def test_graph_store_initialization():
    """Test GraphStore initialization with config."""
    config = BackendConfig(
        backend_type=StorageBackend.GRAPH,
        connection_string="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    store = GraphStore(config)
    assert store.config == config
    store.close()

def test_graph_store_add_simple_node():
    """Test adding a simple node to graph."""
    pytest.skip("Neo4j not available in this environment")
    config = BackendConfig(
        backend_type=StorageBackend.GRAPH,
        connection_string="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    store = GraphStore(config)
    store.clear()

    record = DataRecord(
        id="node1",
        content="Document about machine learning",
        metadata={"dataset": "test", "category": "AI"}
    )
    store.add([record])

    # Verify node exists
    count = store.count()
    assert count == 1

    store.clear()
    store.close()

def test_graph_store_search_by_property():
    """Test property-based search in graph."""
    pytest.skip("Neo4j not available in this environment")
    config = BackendConfig(
        backend_type=StorageBackend.GRAPH,
        connection_string="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    store = GraphStore(config)
    store.clear()

    records = [
        DataRecord(id="n1", content="Alice works at Acme Corp", metadata={"person": "Alice", "org": "Acme"}),
        DataRecord(id="n2", content="Bob works at Beta Inc", metadata={"person": "Bob", "org": "Beta"}),
    ]
    store.add(records)

    # Search by property
    results = store.search(query="", where={"person": "Alice"})
    assert len(results) >= 1
    assert any(r.document_id == "n1" for r in results)

    store.clear()
    store.close()
