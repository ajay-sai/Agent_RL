"""Neo4j graph database backend."""
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable

from ..types import DataRecord, BackendConfig, StorageBackend, RetrievalResult


class GraphStore:
    """Graph database wrapper using Neo4j."""

    def __init__(self, config: BackendConfig):
        """
        Initialize Neo4j connection.

        Args:
            config: BackendConfig with connection details
        """
        if config.backend_type != StorageBackend.GRAPH:
            raise ValueError(f"Expected GRAPH backend, got {config.backend_type}")

        self.config = config
        self._driver: Optional[Driver] = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self._driver = GraphDatabase.driver(
                self.config.connection_string,
                auth=(self.config.username, self.config.password),
            )
            self._driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")

    @property
    def driver(self) -> Driver:
        """Get driver, reconnect if needed."""
        if self._driver is None:
            self._connect()
        return self._driver

    def add(self, records: List[DataRecord]) -> None:
        """
        Add records as nodes to the graph.

        Args:
            records: List of DataRecord objects
        """
        with self.driver.session() as session:
            for record in records:
                query = """
                MERGE (n:Document {id: $id})
                SET n.content = $content
                SET n += $metadata
                """
                session.run(
                    query,
                    id=record.id,
                    content=record.content,
                    metadata=record.metadata or {},
                )

    def search(
        self,
        query: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Search graph by metadata properties or content.

        Args:
            query: Full-text search (reserved for future)
            k: Maximum number of results
            where: Dictionary of property filters

        Returns:
            List of RetrievalResult objects
        """
        with self.driver.session() as session:
            conditions = []
            params = {"k": k}

            if where:
                for key, value in where.items():
                    conditions.append(f"n.{key} = ${key}")
                    params[key] = value

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            cypher = f"""
            MATCH (n:Document)
            {where_clause}
            RETURN n.id as id, n.content as content, n as node
            LIMIT $k
            """

            result = session.run(cypher, **params)

            retrieval_results = []
            for record in result:
                node_props = dict(record["node"].items())
                metadata = {k: v for k, v in node_props.items() if k not in ["id", "content"]}

                retrieval_results.append(RetrievalResult(
                    document_id=record["id"],
                    content=record["content"],
                    score=1.0,
                    source_backend=StorageBackend.GRAPH,
                    metadata=metadata,
                ))

            return retrieval_results

    def clear(self) -> None:
        """Delete all Document nodes from the graph."""
        with self.driver.session() as session:
            session.run("MATCH (n:Document) DETACH DELETE n")

    def count(self) -> int:
        """Count number of Document nodes."""
        with self.driver.session() as session:
            result = session.run("MATCH (n:Document) RETURN count(n) as count")
            record = result.single()
            return record["count"] if record else 0

    def close(self) -> None:
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
