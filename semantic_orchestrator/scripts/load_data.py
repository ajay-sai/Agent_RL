#!/usr/bin/env python
"""Load dataset into storage backends and register in schema registry."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_orchestrator.config import load_config
from semantic_orchestrator.ingestion import CSVLoader, load_csv_dataset
from semantic_orchestrator.registry import SchemaRegistry
from semantic_orchestrator.storage import VectorStore, GraphStore, SQLStore
from semantic_orchestrator.types import BackendConfig, StorageBackend


def assign_backends(schema):
    """
    Determine which storage backend each field/record should go to.

    Simple heuristic:
    - Text-heavy fields (reviews) → Vector
    - Structured measures, IDs → SQL
    - Relationship fields (customer_id, product_category) → Graph
    - All records go to all backends with appropriate filtering
    """
    assignments = {
        StorageBackend.VECTOR: [],  # Will store full documents with text
        StorageBackend.GRAPH: [],   # Will store nodes/relationships
        StorageBackend.SQL: [],     # Will store structured rows
    }

    # For this demo, we'll store all records in all backends but with different metadata emphasis
    # In a real system, you'd split more intelligently
    for field in schema.fields:
        if field.semantic_type == "text":
            assignments[StorageBackend.VECTOR].append(field.name)
        if field.semantic_type in ["identifier", "dimension"]:
            assignments[StorageBackend.GRAPH].append(field.name)
        if field.semantic_type in ["measure", "temporal", "identifier"]:
            assignments[StorageBackend.SQL].append(field.name)

    return assignments


def load_dataset(csv_path: str, dataset_name: str):
    """Load CSV, store in backends, and register schema."""
    config = load_config()

    # Load data
    print(f"Loading dataset from {csv_path}...")
    records = load_csv_dataset(csv_path, dataset_name=dataset_name)
    print(f"Loaded {len(records)} records")

    # Infer schema
    loader = CSVLoader()
    df = loader.load(csv_path)
    schema = loader.infer_schema(df, dataset_name)
    print(f"Inferred schema: {len(schema.fields)} fields")
    for f in schema.fields:
        print(f"  {f.name}: {f.field_type} -> {f.semantic_type}")

    # Assign backends
    assignments = assign_backends(schema)
    print("Backend assignments:")
    for backend, fields in assignments.items():
        print(f"  {backend.value}: {fields}")

    # Initialize storage backends
    stores = {}
    for backend in [StorageBackend.VECTOR, StorageBackend.GRAPH, StorageBackend.SQL]:
        # Map StorageBackend to config field
        config_map = {
            StorageBackend.VECTOR: config.storage.vector,
            StorageBackend.GRAPH: config.storage.graph,
            StorageBackend.SQL: config.storage.sql,
        }
        cfg = config_map[backend]
        backend_config = BackendConfig(
            backend_type=backend,
            connection_string=cfg.persist_directory if backend == StorageBackend.VECTOR else cfg.database,
            collection_name=cfg.collection_name if backend == StorageBackend.VECTOR else None,
            username=cfg.username if hasattr(cfg, 'username') else None,
            password=cfg.password if hasattr(cfg, 'password') else None,
        )
        if backend == StorageBackend.VECTOR:
            stores[backend] = VectorStore(backend_config)
        elif backend == StorageBackend.GRAPH:
            try:
                stores[backend] = GraphStore(backend_config)
            except Exception as e:
                print(f"Warning: GraphStore not available ({e}), skipping")
                stores[backend] = None
        elif backend == StorageBackend.SQL:
            stores[backend] = SQLStore(backend_config)

    # Store records in all backends (for now)
    print("Storing records in backends...")
    for backend, store in stores.items():
        if store is not None:
            try:
                store.add(records)
                count = store.count()
                print(f"  {backend.value}: {count} documents")
            except Exception as e:
                print(f"  {backend.value}: error - {e}")

    # Register schema with backends
    registry = SchemaRegistry()
    assigned_backends = [b for b, store in stores.items() if store is not None]
    registry.register_dataset(schema, backends=assigned_backends)

    # Save registry for later use
    import pickle
    registry_path = Path(config.data.processed_dir) / "registry.pkl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "wb") as f:
        pickle.dump(registry, f)
    print(f"Registry saved to {registry_path}")

    # Close stores
    for store in stores.values():
        if store and hasattr(store, "close"):
            store.close()

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Load dataset into semantic orchestrator storage")
    parser.add_argument("--dataset", type=str, default="sample_sales",
                        help="Dataset name (looks in data/raw/{dataset}.csv)")
    parser.add_argument("--csv", type=str, help="Path to CSV file (overrides --dataset)")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
        dataset_name = Path(csv_path).stem
    else:
        csv_path = f"data/raw/{args.dataset}.csv"
        dataset_name = args.dataset

    load_dataset(csv_path, dataset_name)


if __name__ == "__main__":
    main()
