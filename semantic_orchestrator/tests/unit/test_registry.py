"""Tests for schema registry."""
from semantic_orchestrator.registry import SchemaRegistry
from semantic_orchestrator.types import DatasetSchema, SchemaField, StorageBackend

def test_registry_initialization():
    """Test registry starts empty."""
    registry = SchemaRegistry()
    assert registry.list_datasets() == []

def test_register_dataset():
    """Test registering a dataset with its schema."""
    registry = SchemaRegistry()
    schema = DatasetSchema(
        name="sales",
        fields=[
            SchemaField("date", "object", "temporal", [], False),
            SchemaField("amount", "float64", "measure", [], False),
        ],
        primary_key=["date"]
    )
    registry.register_dataset(schema, backends=[StorageBackend.VECTOR, StorageBackend.SQL])
    assert "sales" in registry.list_datasets()

def test_get_dataset_schema():
    """Test retrieving stored schema."""
    registry = SchemaRegistry()
    schema = DatasetSchema(
        name="customers",
        fields=[
            SchemaField("customer_id", "int64", "identifier", [], False),
            SchemaField("name", "object", "text", [], False),
        ],
    )
    registry.register_dataset(schema, backends=[StorageBackend.VECTOR])
    retrieved = registry.get_dataset_schema("customers")
    assert retrieved.name == "customers"
    assert len(retrieved.fields) == 2

def test_get_backends_for_dataset():
    """Test retrieving backend assignments."""
    registry = SchemaRegistry()
    schema = DatasetSchema(name="products", fields=[])
    registry.register_dataset(schema, backends=[StorageBackend.GRAPH, StorageBackend.SQL])
    backends = registry.get_backends_for_dataset("products")
    assert StorageBackend.GRAPH in backends
    assert StorageBackend.SQL in backends
    assert StorageBackend.VECTOR not in backends

def test_register_overwrites():
    """Test that registering same dataset updates it."""
    registry = SchemaRegistry()
    schema1 = DatasetSchema(name="test", fields=[SchemaField("a", "int64", "measure", [])])
    schema2 = DatasetSchema(name="test", fields=[SchemaField("b", "object", "text", [])])
    registry.register_dataset(schema1, backends=[StorageBackend.VECTOR])
    registry.register_dataset(schema2, backends=[StorageBackend.SQL])
    retrieved = registry.get_dataset_schema("test")
    assert len(retrieved.fields) == 1
    assert retrieved.fields[0].name == "b"
    backends = registry.get_backends_for_dataset("test")
    assert backends == [StorageBackend.SQL]
