"""Tests for core type definitions."""
from dataclasses import asdict
import pytest
from semantic_orchestrator.types import (
    DataRecord,
    SchemaField,
    DatasetSchema,
    StorageBackend,
    RetrievalResult,
    QueryPlan,
    BackendConfig,
)

def test_data_record_creation():
    """Test DataRecord can be created with required fields."""
    record = DataRecord(
        id="rec-001",
        content="Sample text content",
        metadata={"source": "test.csv", "row": 1}
    )
    assert record.id == "rec-001"
    assert record.content == "Sample text content"
    assert record.metadata["source"] == "test.csv"

def test_schema_field_semantic_type():
    """Test SchemaField semantic type inference."""
    field = SchemaField(
        name="customer_id",
        field_type="int64",
        semantic_type="identifier",
        sample_values=[1, 2, 3],
        nullable=False
    )
    assert field.semantic_type == "identifier"

def test_dataset_schema_serialization():
    """Test DatasetSchema can be serialized to dict."""
    schema = DatasetSchema(
        name="sales",
        fields=[
            SchemaField("date", "datetime64", "temporal", ["2024-01-01"], False),
            SchemaField("amount", "float64", "measure", [100.0, 200.0], True),
        ],
        primary_key=["date"]
    )
    d = asdict(schema)
    assert d["name"] == "sales"
    assert len(d["fields"]) == 2

def test_storage_backend_enum():
    """Test StorageBackend enum values."""
    assert StorageBackend.VECTOR == "vector"
    assert StorageBackend.GRAPH == "graph"
    assert StorageBackend.SQL == "sql"
    assert StorageBackend.ALL == "all"

def test_retrieval_result_confidence():
    """Test RetrievalResult stores confidence score."""
    result = RetrievalResult(
        document_id="doc-1",
        content="Answer text",
        score=0.95,
        source_backend="vector",
        metadata={}
    )
    assert result.score == 0.95
    assert 0 <= result.score <= 1

def test_query_plan_validates_backends():
    """Test QueryPlan validates backend choices."""
    plan = QueryPlan(
        query="test query",
        backends=[StorageBackend.VECTOR, StorageBackend.GRAPH],
        weights={"vector": 0.7, "graph": 0.3}
    )
    assert len(plan.backends) == 2
    assert sum(plan.weights.values()) == 1.0

def test_backend_config_needed_fields():
    """Test BackendConfig requires connection details."""
    config = BackendConfig(
        backend_type=StorageBackend.VECTOR,
        connection_string="path/to/chroma",
        collection_name="test"
    )
    assert config.backend_type == StorageBackend.VECTOR
