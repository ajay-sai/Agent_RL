"""Tests for data ingestion module."""
import pytest
import pandas as pd
from pathlib import Path
from semantic_orchestrator.ingestion import CSVLoader, load_csv_dataset

def test_csv_loader_basic():
    """Test CSVLoader can read a simple CSV."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,name,amount\n")
        f.write("1,Alice,100.5\n")
        f.write("2,Bob,200.0\n")
        csv_path = f.name

    try:
        loader = CSVLoader()
        df = loader.load(csv_path)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "amount"]
    finally:
        Path(csv_path).unlink()

def test_csv_loader_infers_schema():
    """Test loader infers data types correctly."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,date,amount,category\n")
        f.write("1,2024-01-15,100.50,Electronics\n")
        f.write("2,2024-01-16,200.00,Books\n")
        csv_path = f.name

    try:
        loader = CSVLoader()
        df = loader.load(csv_path)
        # Check inferred dtypes
        assert df["id"].dtype in ["int64", "Int64"]
        assert pd.api.types.is_datetime64_any_dtype(df["date"]) or df["date"].dtype == "object"
        assert df["amount"].dtype in ["float64", "Float64"]
        assert df["category"].dtype == "object"
    finally:
        Path(csv_path).unlink()

def test_csv_loader_semantic_inference():
    """Test semantic type inference."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("customer_id,revenue,product_category,review_text\n")
        f.write("1,1500.00,Electronics,Great product!\n")
        f.write("2,2500.50,Books,Very informative.\n")
        csv_path = f.name

    try:
        loader = CSVLoader()
        df = loader.load(csv_path)
        schema = loader.infer_schema(df, "test")
        fields = {f.name: f for f in schema.fields}

        # customer_id should be identifier
        assert fields["customer_id"].semantic_type == "identifier"
        # revenue should be measure
        assert fields["revenue"].semantic_type == "measure"
        # product_category should be dimension (low cardinality categorical)
        assert fields["product_category"].semantic_type == "dimension"
        # review_text should be text (high cardinality)
        assert fields["review_text"].semantic_type == "text"
    finally:
        Path(csv_path).unlink()

def test_load_csv_dataset_convenience():
    """Test the convenience function load_csv_dataset."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,text\n")
        f.write("1,Hello world\n")
        f.write("2,Test document\n")
        csv_path = f.name

    try:
        records = load_csv_dataset(csv_path, dataset_name="test")
        assert len(records) == 2
        assert records[0].id == "1"
        # Content includes all fields as "field: value" pairs
        assert "Hello world" in records[0].content
        assert records[0].metadata["dataset"] == "test"
        assert records[0].metadata["source"] == "csv"
    finally:
        Path(csv_path).unlink()

def test_csv_loader_to_records():
    """Test conversion of DataFrame to DataRecords."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,value,category\n")
        f.write("101,42.5,A\n")
        f.write("102,43.1,B\n")
        csv_path = f.name

    try:
        loader = CSVLoader()
        df = loader.load(csv_path)
        records = loader.to_records(df, "test_data")
        assert len(records) == 2
        # Check that all row fields appear in metadata
        for rec in records:
            assert "value" in rec.metadata
            assert "category" in rec.metadata
    finally:
        Path(csv_path).unlink()
