"""SQLite storage backend for structured queries."""
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, Column, String, Text, JSON, Integer, MetaData, Table
from sqlalchemy.orm import sessionmaker

from ..types import DataRecord, BackendConfig, StorageBackend, RetrievalResult


class SQLStore:
    """SQL storage wrapper using SQLite."""

    def __init__(self, config: BackendConfig):
        """
        Initialize SQL store.

        Args:
            config: BackendConfig with database path
        """
        if config.backend_type != StorageBackend.SQL:
            raise ValueError(f"Expected SQL backend, got {config.backend_type}")

        self.config = config
        db_path = Path(config.connection_string)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create SQLAlchemy engine
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=config.extra_params.get("echo", False),
            future=True
        )

        # Define table schema
        self.metadata = MetaData()
        self.table = Table(
            "documents",
            self.metadata,
            Column("id", String, primary_key=True),
            Column("content", Text),
            Column("metadata", JSON),
        )

        # Create table if not exists
        self.metadata.create_all(self.engine)

        self.Session = sessionmaker(bind=self.engine)

    def add(self, records: List[DataRecord]) -> None:
        """
        Insert records into SQL table.

        Args:
            records: List of DataRecord objects
        """
        if not records:
            return

        with self.Session() as session:
            for record in records:
                stmt = self.table.insert().values(
                    id=record.id,
                    content=record.content,
                    metadata=record.metadata or {},
                )
                session.execute(stmt)
            session.commit()

    def search(
        self,
        query: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Search documents by metadata filters.

        Args:
            query: Full-text search (basic LIKE for now)
            k: Maximum results
            where: Metadata filters

        Returns:
            List of RetrievalResult
        """
        with self.Session() as session:
            stmt = self.table.select()

            # Apply metadata filters
            if where:
                for key, value in where.items():
                    stmt = stmt.where(self.table.c.metadata[key] == value)

            # text search if query provided
            if query:
                stmt = stmt.where(self.table.c.content.like(f"%{query}%"))

            stmt = stmt.limit(k)
            result = session.execute(stmt)

            retrieval_results = []
            for row in result:
                metadata = dict(row.metadata or {})
                retrieval_results.append(RetrievalResult(
                    document_id=row.id,
                    content=row.content,
                    score=1.0,
                    source_backend=StorageBackend.SQL,
                    metadata=metadata,
                ))

            return retrieval_results

    def clear(self) -> None:
        """Delete all rows from documents table."""
        with self.Session() as session:
            stmt = self.table.delete()
            session.execute(stmt)
            session.commit()

    def count(self) -> int:
        """Count documents in table."""
        with self.Session() as session:
            from sqlalchemy import func
            result = session.query(func.count(self.table.c.id)).scalar()
            return result or 0

    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()
