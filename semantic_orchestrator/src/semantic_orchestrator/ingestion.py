"""Data ingestion module for loading and preprocessing datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional

import pandas as pd
from pydantic import BaseModel

from .types import DataRecord, DatasetSchema, SchemaField


class CSVLoader:
    """Loads CSV files with intelligent schema inference."""

    def __init__(self, encoding: str = "utf-8", delimiter: str = ","):
        self.encoding = encoding
        self.delimiter = delimiter

    def load(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.

        Args:
            path: Path to CSV file
            **kwargs: Additional arguments to pd.read_csv

        Returns:
            pandas DataFrame
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(
            path,
            encoding=self.encoding,
            delimiter=self.delimiter,
            **kwargs
        )
        return df

    def infer_schema(self, df: pd.DataFrame, dataset_name: str) -> DatasetSchema:
        """
        Infer schema from DataFrame.

        Args:
            df: pandas DataFrame
            dataset_name: Name for this dataset

        Returns:
            DatasetSchema object
        """
        fields = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].dropna().head(5).tolist()
            semantic_type = self._guess_semantic_type(df[col], sample_values)

            field = SchemaField(
                name=col,
                field_type=dtype,
                semantic_type=semantic_type,
                sample_values=sample_values,
                nullable=df[col].isnull().any(),
            )
            fields.append(field)

        # Guess primary key
        primary_key = None
        id_cols = [f.name for f in fields if f.name.lower() == "id"]
        if id_cols:
            primary_key = id_cols
        else:
            unique_cols = [
                f.name for f in fields
                if f.name in df.columns and df[f.name].is_unique and "int" in f.field_type
            ]
            if unique_cols:
                primary_key = unique_cols[:1]

        return DatasetSchema(
            name=dataset_name,
            fields=fields,
            primary_key=primary_key,
        )

    def _guess_semantic_type(self, series: pd.Series, samples: List[Any]) -> str:
        """Heuristic to guess semantic type from data."""
        dtype = str(series.dtype)
        col_name = series.name.lower() if series.name else ""

        # Temporal
        if "datetime" in dtype or "date" in col_name:
            return "temporal"

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            if any(keyword in col_name for keyword in ["id", "key", "code"]):
                return "identifier"
            return "measure"

        # Text / Categorical
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            n_unique = series.nunique()
            # Text indicators in name
            if any(keyword in col_name for keyword in ["text", "content", "description", "comment", "review", "body", "message"]):
                return "text"
            # Categorical if: name suggests category OR low unique count (<10) OR low ratio
            if any(keyword in col_name for keyword in ["category", "type", "status", "region", "country", "city", "state", "gender"]):
                return "dimension"
            elif n_unique < 10:
                return "dimension"
            else:
                unique_ratio = n_unique / len(series) if len(series) > 0 else 0
                if unique_ratio < 0.1:
                    return "dimension"
                else:
                    return "text"

        # Boolean
        if pd.api.types.is_bool_dtype(series):
            return "dimension"

        return "text"

    def to_records(self, df: pd.DataFrame, dataset_name: str) -> List[DataRecord]:
        """
        Convert DataFrame to list of DataRecords.

        Args:
            df: DataFrame to convert
            dataset_name: Name to attach as metadata

        Returns:
            List of DataRecord objects
        """
        records = []
        for idx, row in df.iterrows():
            # Use the 'id' column if present, otherwise index
            record_id = str(row.get("id", idx)) if "id" in df.columns else str(idx)
            content = " | ".join(f"{col}: {val}" for col, val in row.items())

            metadata = {
                "dataset": dataset_name,
                "source": "csv",
                "row_index": int(idx),
            }
            metadata.update(row.to_dict())

            record = DataRecord(
                id=record_id,
                content=content,
                metadata=metadata,
            )
            records.append(record)

        return records


def load_csv_dataset(
    path: str | Path,
    dataset_name: Optional[str] = None,
    loader: Optional[CSVLoader] = None,
) -> List[DataRecord]:
    """
    Convenience function to load CSV and convert to DataRecords.

    Args:
        path: Path to CSV file
        dataset_name: Name for dataset (defaults to filename without extension)
        loader: CSVLoader instance

    Returns:
        List of DataRecord objects
    """
    if dataset_name is None:
        dataset_name = Path(path).stem

    if loader is None:
        loader = CSVLoader()

    df = loader.load(path)
    records = loader.to_records(df, dataset_name)
    return records
