"""Schema registry that tracks datasets and their storage assignments."""
from typing import Dict, List, Optional

from .types import DatasetSchema, StorageBackend


class SchemaRegistry:
    """Central registry for dataset schemas and backend assignments."""

    def __init__(self):
        """Initialize an empty registry."""
        # Map: dataset_name -> DatasetSchema
        self._schemas: Dict[str, DatasetSchema] = {}
        # Map: dataset_name -> List[StorageBackend]
        self._backend_assignments: Dict[str, List[StorageBackend]] = {}

    def register_dataset(
        self,
        schema: DatasetSchema,
        backends: List[StorageBackend]
    ) -> None:
        """
        Register a dataset with its schema and assigned storage backends.

        Args:
            schema: DatasetSchema object
            backends: List of StorageBackend values where this dataset is stored
        """
        if not backends:
            raise ValueError("At least one backend must be specified")

        self._schemas[schema.name] = schema
        self._backend_assignments[schema.name] = list(backends)

    def get_dataset_schema(self, name: str) -> Optional[DatasetSchema]:
        """
        Retrieve schema for a dataset.

        Args:
            name: Dataset name

        Returns:
            DatasetSchema if found, else None
        """
        return self._schemas.get(name)

    def get_backends_for_dataset(self, name: str) -> List[StorageBackend]:
        """
        Get which storage backends contain this dataset.

        Args:
            name: Dataset name

        Returns:
            List of StorageBackend values (empty if dataset not registered)
        """
        return self._backend_assignments.get(name, [])

    def list_datasets(self) -> List[str]:
        """
        List all registered dataset names.

        Returns:
            List of dataset names
        """
        return list(self._schemas.keys())

    def is_available_on_backend(self, dataset_name: str, backend: StorageBackend) -> bool:
        """
        Check if a dataset is available on a specific backend.

        Args:
            dataset_name: Name of dataset
            backend: Storage backend to check

        Returns:
            True if dataset is stored on that backend
        """
        return backend in self._backend_assignments.get(dataset_name, [])
