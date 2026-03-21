"""Integration tests for the semantic orchestrator pipeline."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.semantic_orchestrator import (
    QueryOrchestrator,
    SynthesisAgent,
    RouterAgent,
    SchemaRegistry,
    BackendConfig,
    StorageBackend,
    DataRecord,
)
from src.semantic_orchestrator.config import Config
from src.semantic_orchestrator.retrieval import create_retriever_for_backend
from src.semantic_orchestrator.storage import VectorStore, SQLStore


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary configuration for testing."""
    config = Config()

    # Override paths to use temporary directory
    config.storage.vector.persist_directory = str(tmp_path / "chroma")
    config.storage.sql.database = str(tmp_path / "semantic.db")
    config.data.processed_dir = str(tmp_path / "processed")
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.embeddings_cache = str(tmp_path / "embeddings_cache")

    # Ensure directories exist
    Path(config.storage.vector.persist_directory).mkdir(parents=True, exist_ok=True)
    Path(config.data.processed_dir).mkdir(parents=True, exist_ok=True)

    return config


@pytest.fixture
def sample_data():
    """Return sample DataRecords for testing."""
    return [
        DataRecord(
            id="doc1",
            content="Machine learning is a subset of artificial intelligence",
            metadata={"topic": "ML", "category": "AI"}
        ),
        DataRecord(
            id="doc2",
            content="Deep learning uses neural networks with many layers",
            metadata={"topic": "DL", "category": "AI"}
        ),
        DataRecord(
            id="doc3",
            content="Python is a popular programming language for data science",
            metadata={"topic": "programming", "category": "language"}
        ),
    ]


def test_vector_store_integration(temp_config, sample_data):
    """Test vector store can store and retrieve."""
    backend_config = BackendConfig(
        backend_type=StorageBackend.VECTOR,
        connection_string=temp_config.storage.vector.persist_directory,
        collection_name="test_integration",
        extra_params={"embedding_model": "all-MiniLM-L6-v2"}
    )
    store = VectorStore(backend_config)
    store.add(sample_data)
    assert store.count() == 3

    results = store.search("neural networks", k=2)
    assert len(results) >= 1
    # Should find doc2 about deep learning
    assert any(r.document_id == "doc2" for r in results)

    store.clear()
    store.close()


def test_sql_store_integration(temp_config, sample_data):
    """Test SQL store can store and retrieve."""
    db_path = Path(temp_config.storage.sql.database)
    backend_config = BackendConfig(
        backend_type=StorageBackend.SQL,
        connection_string=str(db_path),
    )
    store = SQLStore(backend_config)
    store.add(sample_data)
    assert store.count() == 3

    results = store.search("Python", k=2)
    assert len(results) >= 1
    assert any("Python" in r.content for r in results)

    store.clear()
    store.close()


def test_registry_persistence(temp_config):
    """Test registry can be saved and loaded."""
    registry = SchemaRegistry()
    schema = MagicMock()
    schema.name = "test_dataset"
    registry.register_dataset(schema, backends=[StorageBackend.VECTOR, StorageBackend.SQL])

    # Save
    registry_path = Path(temp_config.data.processed_dir) / "registry.pkl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(registry_path, "wb") as f:
        pickle.dump(registry, f)

    # Load
    with open(registry_path, "rb") as f:
        loaded = pickle.load(f)

    assert "test_dataset" in loaded.list_datasets()
    assert StorageBackend.VECTOR in loaded.get_backends_for_dataset("test_dataset")


def test_router_decides_valid_backend():
    """Test router returns valid backends when registry has data."""
    registry = SchemaRegistry()
    schema = MagicMock()
    schema.name = "test"
    registry.register_dataset(schema, backends=[StorageBackend.VECTOR])

    router = RouterAgent(registry=registry, device="cpu")
    plan = router.decide("test query", eval_mode=True)

    assert plan.backends is not None
    # Should only choose backends that have registered datasets
    for backend in plan.backends:
        assert backend in [StorageBackend.VECTOR]  # Only VECTOR has data


def test_retriever_coordination(temp_config, sample_data):
    """Test that multiple retrievers can be created and used together."""
    # Setup stores
    vector_backend = BackendConfig(
        backend_type=StorageBackend.VECTOR,
        connection_string=temp_config.storage.vector.persist_directory,
        collection_name="coord_test",
    )
    sql_backend = BackendConfig(
        backend_type=StorageBackend.SQL,
        connection_string=temp_config.storage.sql.database,
    )

    # Add same data to both
    VectorStore(vector_backend).add(sample_data)
    SQLStore(sql_backend).add(sample_data)

    # Create retrievers
    vector_retriever = create_retriever_for_backend(StorageBackend.VECTOR, vector_backend)
    sql_retriever = create_retriever_for_backend(StorageBackend.SQL, sql_backend)

    # Search both
    vector_results = vector_retriever.search("neural", k=2)
    sql_results = sql_retriever.search("Python", k=2)

    assert isinstance(vector_results, list)
    assert isinstance(sql_results, list)

    # Cleanup
    vector_retriever.close()
    sql_retriever.close()


def test_synthesis_with_results():
    """Test synthesis agent processes retrieval results."""
    # This test requires OPENROUTER_API_KEY, so skip if not set
    import os
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    from src.semantic_orchestrator.synthesis import SynthesisAgent

    agent = SynthesisAgent()
    results = [
        DataRecord(
            id="doc1",
            content="Machine learning content",
            metadata={}
        )
    ]
    # Mock RetrievalResult objects
    from src.semantic_orchestrator.types import RetrievalResult as RR
    retrieval_results = [
        RR(document_id="doc1", content="Machine learning is AI", score=0.9, source_backend=StorageBackend.VECTOR, metadata={}),
        RR(document_id="doc2", content="Deep learning uses neural nets", score=0.8, source_backend=StorageBackend.SQL, metadata={}),
    ]

    answer = agent.process("What is ML?", retrieval_results, top_k=2)
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_full_pipeline_mocked(temp_config, sample_data):
    """Test end-to-end pipeline with mocked components where needed."""
    # This is a smoke test that verifies the plumbing works
    # We'll use real stores but mock the synthesis LLM call

    # 1. Prepare data
    from src.semantic_orchestrator.ingestion import CSVLoader
    import pandas as pd

    # Create a simple CSV
    csv_path = temp_config.data.raw_dir + "/test.csv"
    Path(temp_config.data.raw_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {"id": 1, "content": "ML content", "category": "AI"},
        {"id": 2, "content": "DL content", "category": "AI"},
    ])
    df.to_csv(csv_path, index=False)

    # Load into vector store
    loader = CSVLoader()
    records = loader.to_records(df, "test")
    vector_store = VectorStore(BackendConfig(
        backend_type=StorageBackend.VECTOR,
        connection_string=temp_config.storage.vector.persist_directory,
        collection_name="test",
    ))
    vector_store.add(records)
    assert vector_store.count() == 2

    # Create registry
    registry = SchemaRegistry()
    from src.semantic_orchestrator.types import DatasetSchema, SchemaField
    schema = DatasetSchema(
        name="test",
        fields=[
            SchemaField("id", "int64", "identifier", []),
            SchemaField("content", "object", "text", []),
            SchemaField("category", "object", "dimension", []),
        ],
        primary_key=["id"]
    )
    registry.register_dataset(schema, backends=[StorageBackend.VECTOR])

    # Save registry
    registry_path = Path(temp_config.data.processed_dir) / "registry.pkl"
    import pickle
    with open(registry_path, "wb") as f:
        pickle.dump(registry, f)

    # 2. Test QueryOrchestrator
    # We need to mock the synthesis agent's LLM call
    with patch.object(SynthesisAgent, 'process', return_value="Mocked answer"):
        orch = QueryOrchestrator(config_path=None)
        # Override config with our temp config
        orch.config = temp_config
        orch.registry = registry
        orch.router = RouterAgent(registry=registry, device="cpu")

        answer = orch.query("test query")
        assert answer == "Mocked answer"

    # Cleanup
    vector_store.clear()
    vector_store.close()
