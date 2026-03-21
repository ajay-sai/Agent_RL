"""Query orchestration - coordinates the end-to-end query pipeline."""
import logging
from pathlib import Path
from typing import List, Optional

from .config import load_config
from .registry import SchemaRegistry
from .router import RouterAgent
from .retrieval import create_retriever_for_backend
from .synthesis import SynthesisAgent
from .types import RetrievalResult, BackendConfig, StorageBackend

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the query pipeline: routing → retrieval → synthesis."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the orchestrator.

        Args:
            config_path: Optional path to config.yaml

        Raises:
            FileNotFoundError: If registry.pkl not found
            ValueError: If no backends are available
        """
        self.config = load_config(config_path)

        # Load registry
        registry_path = Path(self.config.data.processed_dir) / "registry.pkl"
        if not registry_path.exists():
            raise FileNotFoundError(
                f"Registry not found at {registry_path}. "
                "Please run 'semantic-orchestrator load-data' first to load data into storage backends."
            )
        import pickle
        with open(registry_path, "rb") as f:
            self.registry = pickle.load(f)
        logger.info(f"Loaded registry with datasets: {self.registry.list_datasets()}")

        # Initialize router
        self.router = RouterAgent(config_path=config_path, registry=self.registry)

        # Initialize synthesis agent
        self.synthesis_agent = SynthesisAgent(config_path=config_path)

        # Keep track of retrievers we create (for cleanup)
        self._retrievers: List = []

    def _create_retriever(self, backend: StorageBackend):
        """Create a retriever for the given backend."""
        # Build BackendConfig from global config
        config_map = {
            StorageBackend.VECTOR: self.config.storage.vector,
            StorageBackend.GRAPH: self.config.storage.graph,
            StorageBackend.SQL: self.config.storage.sql,
        }
        cfg = config_map[backend]
        backend_config = BackendConfig(
            backend_type=backend,
            connection_string=cfg.persist_directory if backend == StorageBackend.VECTOR else cfg.database,
            collection_name=cfg.collection_name if backend == StorageBackend.VECTOR else None,
            username=getattr(cfg, 'username', None),
            password=getattr(cfg, 'password', None),
        )
        return create_retriever_for_backend(backend, backend_config)

    def query(self, query_text: str, top_k: int = 10) -> str:
        """
        Execute a query and return the synthesized answer.

        Args:
            query_text: User's natural language query
            top_k: Number of results to retrieve from each backend

        Returns:
            Synthesized answer string

        Raises:
            ValueError: If no backends are available or query fails
        """
        logger.info(f"Processing query: {query_text}")

        # Get available datasets
        available_datasets = self.router.registry.list_datasets()
        if not available_datasets:
            raise ValueError("No datasets registered. Please load data first.")

        # Router decides which backends to use
        plan = self.router.decide(query_text, available_datasets=available_datasets, eval_mode=True)
        logger.info(f"Router selected backends: {[b.value for b in plan.backends]}")

        if not plan.backends:
            raise ValueError("Router returned no backends. Check that data is loaded on at least one backend.")

        # Execute retrievals
        all_results: List[RetrievalResult] = []
        self._retrievers = []

        for backend in plan.backends:
            try:
                retriever = self._create_retriever(backend)
                self._retrievers.append(retriever)
                results = retriever.search(query_text, k=top_k)
                logger.info(f"Retrieved {len(results)} results from {backend.value}")
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to retrieve from {backend.value}: {e}")
                continue

        if not all_results:
            raise ValueError("No results retrieved from any backend.")

        # Synthesize answer
        try:
            answer = self.synthesis_agent.process(
                query_text,
                all_results,
                deduplicate=True,
                rerank=True,
                top_k=top_k
            )
            logger.info("Synthesis complete")
            return answer
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise ValueError(f"Failed to generate answer: {e}")

    def close(self):
        """Clean up resources (close retriever connections)."""
        for retriever in self._retrievers:
            try:
                retriever.close()
            except Exception:
                pass
        self._retrievers = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
