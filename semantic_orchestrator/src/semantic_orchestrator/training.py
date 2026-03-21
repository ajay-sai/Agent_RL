"""RL training loop for the RouterAgent."""
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
import torch

from .config import load_config
from .router import RouterAgent
from .registry import SchemaRegistry
from .retrieval import create_retriever_for_backend
from .storage import VectorStore, GraphStore, SQLStore
from .types import BackendConfig, StorageBackend

logger = logging.getLogger(__name__)


def load_queries(queries_path: Path) -> List[Dict]:
    """Load queries from JSONL file."""
    queries = []
    with open(queries_path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def compute_f1(
    retrieved_ids: Set[str],
    relevant_ids: Set[str]
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        retrieved_ids: Set of retrieved document IDs
        relevant_ids: Set of relevant document IDs (ground truth)

    Returns:
        (precision, recall, f1)
    """
    if not retrieved_ids and not relevant_ids:
        return 1.0, 1.0, 1.0
    if not retrieved_ids:
        return 0.0, 0.0, 0.0
    if not relevant_ids:
        return 0.0, 0.0, 0.0

    tp = len(retrieved_ids & relevant_ids)
    precision = tp / len(retrieved_ids)
    recall = tp / len(relevant_ids)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def execute_query(
    router: RouterAgent,
    query: str,
    top_k: int,
    config
) -> Set[str]:
    """
    Execute a query through the router and retrievers, return set of retrieved doc IDs.

    Args:
        router: RouterAgent instance
        query: Query string
        top_k: Number of results per backend
        config: Config object

    Returns:
        Set of retrieved document IDs
    """
    # Get available datasets
    available_datasets = router.registry.list_datasets()
    if not available_datasets:
        return set()

    # Router decision
    plan = router.decide(query, available_datasets=available_datasets, eval_mode=True)

    # Create retrievers and execute
    all_ids = set()
    retrievers = []

    try:
        for backend in plan.backends:
            # Build BackendConfig
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
                username=getattr(cfg, 'username', None),
                password=getattr(cfg, 'password', None),
            )
            retriever = create_retriever_for_backend(backend, backend_config)
            retrievers.append(retriever)

            results = retriever.search(query, k=top_k)
            all_ids.update(r.document_id for r in results)
    except Exception as e:
        logger.warning(f"Error executing query: {e}")
    finally:
        for retriever in retrievers:
            try:
                retriever.close()
            except:
                pass

    return all_ids


def train_router(
    train_queries: List[Dict],
    val_queries: List[Dict],
    router: RouterAgent,
    config,
    num_epochs: int = 10,
    top_k: int = 10,
    checkpoint_dir: Optional[Path] = None,
    use_wandb: bool = False
):
    """
    Train the router using REINFORCE algorithm.

    Args:
        train_queries: List of training query dicts
        val_queries: List of validation query dicts
        router: RouterAgent instance
        config: Config object with RL settings
        num_epochs: Number of training epochs
        top_k: Number of retrieval results per backend
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to log to wandb
    """
    if use_wandb and config.logging.wandb_project:
        try:
            import wandb
            wandb.init(project=config.logging.wandb_project, entity=config.logging.wandb_entity)
            wandb_enabled = True
        except ImportError:
            logger.warning("wandb not installed, logging disabled")
            wandb_enabled = False
    else:
        wandb_enabled = False

    if checkpoint_dir is None:
        checkpoint_dir = Path(config.logging.log_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Training phase
        router.optimizer.zero_grad()
        epoch_rewards = []
        epoch_losses = []

        random.shuffle(train_queries)

        for query_data in train_queries:
            query = query_data["query"]
            relevant_ids = set(str(id) for id in query_data["relevant_doc_ids"])

            # Execute query (router will store log_probs during decide)
            retrieved_ids = execute_query(router, query, top_k, config)

            # Compute reward (F1 score)
            precision, recall, f1 = compute_f1(retrieved_ids, relevant_ids)
            reward = f1 * config.rl.reward_scale

            # Record reward
            router.reward(reward)
            epoch_rewards.append(f1)

        # Perform training step
        loss = router.train_step()
        epoch_losses.append(loss)

        avg_train_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        logger.info(f"Epoch {epoch + 1}: Train F1 = {avg_train_reward:.4f}, Loss = {loss:.4f}")

        # Validation phase
        if val_queries:
            val_f1_scores = []
            for query_data in val_queries:
                query = query_data["query"]
                relevant_ids = set(str(id) for id in query_data["relevant_doc_ids"])
                retrieved_ids = execute_query(router, query, top_k, config)
                _, _, f1 = compute_f1(retrieved_ids, relevant_ids)
                val_f1_scores.append(f1)

            avg_val_f1 = np.mean(val_f1_scores) if val_f1_scores else 0.0
            logger.info(f"Epoch {epoch + 1}: Val F1 = {avg_val_f1:.4f}")

            # Save best checkpoint
            if avg_val_f1 > best_val_f1:
                best_val_f1 = avg_val_f1
                checkpoint_path = checkpoint_dir / "router_best.pt"
                router.save(checkpoint_path)
                logger.info(f"Saved best checkpoint with Val F1 = {best_val_f1:.4f}")

        # Save latest checkpoint
        checkpoint_path = checkpoint_dir / f"router_epoch_{epoch + 1}.pt"
        router.save(checkpoint_path)

        # Wandb logging
        if wandb_enabled:
            log_dict = {
                "epoch": epoch + 1,
                "train_f1": avg_train_reward,
                "train_loss": loss,
            }
            if val_queries:
                log_dict["val_f1"] = avg_val_f1
            wandb.log(log_dict)

    logger.info(f"Training complete. Best Val F1: {best_val_f1:.4f}")


def main():
    """Main entry point for training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train the router agent")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--top-k", type=int, default=10, help="Top k results per backend")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override LR if provided
    if args.lr:
        config.rl.learning_rate = args.lr

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load queries
    train_queries = load_queries(Path(config.data.queries_train))
    val_queries = load_queries(Path(config.data.queries_val))

    logger.info(f"Loaded {len(train_queries)} train queries, {len(val_queries)} val queries")

    # Initialize router with registry
    registry = SchemaRegistry()
    registry_path = Path(config.data.processed_dir) / "registry.pkl"
    if registry_path.exists():
        import pickle
        with open(registry_path, "rb") as f:
            registry = pickle.load(f)
        logger.info(f"Loaded registry with datasets: {registry.list_datasets()}")
    else:
        logger.warning("Registry not found. Router may not have valid backend assignments.")

    router = RouterAgent(config_path=args.config, registry=registry)

    # Train
    train_router(
        train_queries,
        val_queries,
        router,
        config,
        num_epochs=args.epochs,
        top_k=args.top_k,
        use_wandb=args.wandb
    )


if __name__ == "__main__":
    main()
