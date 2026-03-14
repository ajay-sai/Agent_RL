"""Core type definitions for the semantic orchestrator."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class StorageBackend(str, Enum):
    """Available storage backends."""
    VECTOR = "vector"
    GRAPH = "graph"
    SQL = "sql"
    ALL = "all"  # For router output meaning all backends


@dataclass
class DataRecord:
    """A single data record after ingestion."""
    id: str
    content: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SchemaField:
    """Definition of a field in a dataset schema."""
    name: str
    field_type: str  # e.g., "int64", "float64", "object", "datetime64"
    semantic_type: str  # e.g., "identifier", "measure", "dimension", "text"
    sample_values: List[Any] = None
    nullable: bool = True
    description: Optional[str] = None

    def __post_init__(self):
        if self.sample_values is None:
            self.sample_values = []


@dataclass
class DatasetSchema:
    """Complete schema for a dataset."""
    name: str
    fields: List[SchemaField]
    primary_key: Optional[List[str]] = None
    foreign_keys: Optional[Dict[str, str]] = None  # {field: referenced_dataset}
    description: Optional[str] = None


@dataclass
class BackendConfig:
    """Configuration for a storage backend."""
    backend_type: StorageBackend
    connection_string: str
    collection_name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class RetrievalResult:
    """Result from a retriever agent."""
    document_id: str
    content: str
    score: float  # 0-1 relevance score
    source_backend: StorageBackend
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryPlan:
    """Router's decision about which backends to query."""
    query: str
    backends: List[StorageBackend]
    weights: Optional[Dict[StorageBackend, float]] = None  # For multi-backend fusion
    query_embedding: Optional[List[float]] = None
    reasoning: Optional[str] = None
