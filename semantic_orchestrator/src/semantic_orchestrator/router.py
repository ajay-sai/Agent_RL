"""RL Router Agent - learns to select backend(s) for queries."""
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .config import load_config
from .registry import SchemaRegistry
from .types import QueryPlan, StorageBackend, BackendConfig, RetrievalResult
from .retrieval.retrievers import create_retriever_for_backend

# Action space: 7 discrete combinations
ACTIONS = [
    [StorageBackend.VECTOR],
    [StorageBackend.GRAPH],
    [StorageBackend.SQL],
    [StorageBackend.VECTOR, StorageBackend.GRAPH],
    [StorageBackend.VECTOR, StorageBackend.SQL],
    [StorageBackend.GRAPH, StorageBackend.SQL],
    [StorageBackend.ALL],
]
N_ACTIONS = len(ACTIONS)


class RouterPolicy(nn.Module):
    """Neural network policy that maps query embedding to action logits."""

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128, n_actions: int = N_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for each action."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class RouterAgent:
    """Router that decides which backend(s) to use for a query."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        registry: Optional[SchemaRegistry] = None,
        device: str = "cpu"
    ):
        """
        Initialize router agent.

        Args:
            config_path: Path to config.yaml
            registry: SchemaRegistry instance (loads from saved file if None)
            device: 'cpu' or 'cuda'
        """
        self.config = load_config(config_path)
        self.registry = registry or self._load_saved_registry()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize embedding model (lazy)
        self._embedder: Optional[SentenceTransformer] = None

        # Policy network
        self.policy = RouterPolicy().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.rl.learning_rate
        )

        # For storing trajectory
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embedder is None:
            model_name = self.config.storage.vector.embedding_model
            self._embedder = SentenceTransformer(model_name)
        return self._embedder

    def embed_query(self, query: str) -> torch.Tensor:
        """Embed query string to tensor."""
        embedding = self.embedder.encode(query, convert_to_tensor=True)
        return embedding.to(self.device)

    def _mask_invalid_actions(self, available_datasets: List[str]) -> torch.Tensor:
        """
        Create mask for actions that use backends with no registered datasets.

        Args:
            available_datasets: List of dataset names that have data

        Returns:
            Boolean mask tensor of shape (n_actions,), True = valid
        """
        if not available_datasets:
            # No datasets registered, allow all actions
            return torch.ones(N_ACTIONS, dtype=torch.bool, device=self.device)

        mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=self.device)
        for i, backends in enumerate(ACTIONS):
            # Action is valid if all its backends have at least one dataset
            valid = all(
                any(
                    self.registry.is_available_on_backend(ds, be)
                    for ds in available_datasets
                )
                for be in backends
            )
            mask[i] = valid
        return mask

    def decide(
        self,
        query: str,
        available_datasets: Optional[List[str]] = None,
        eval_mode: bool = False
    ) -> QueryPlan:
        """
        Decide which backend(s) to query.

        Args:
            query: User query string
            available_datasets: List of dataset names to consider (None = all)
            eval_mode: If True, use argmax and no gradient; else sample for exploration

        Returns:
            QueryPlan with selected backends
        """
        if available_datasets is None:
            available_datasets = self.registry.list_datasets()

        # Get query embedding
        embedding = self.embed_query(query).unsqueeze(0)  # (1, embed_dim)

        # Get action logits
        if eval_mode:
            self.policy.eval()
            with torch.no_grad():
                logits = self.policy(embedding)  # (1, n_actions)
        else:
            self.policy.train()
            logits = self.policy(embedding)  # (1, n_actions), requires grad

        # Mask invalid actions (backends with no data)
        mask = self._mask_invalid_actions(available_datasets).unsqueeze(0)  # (1, n_actions)
        logits[~mask] = -torch.inf

        # Sample or take best
        probs = F.softmax(logits, dim=-1)
        if eval_mode:
            action_idx = torch.argmax(probs).item()
        else:
            action_idx = torch.multinomial(probs, num_samples=1).item()

        selected_backends = ACTIONS[action_idx]

        # Store log probability for training (only if training mode)
        if not eval_mode:
            log_prob = F.log_softmax(logits, dim=-1)[0, action_idx]
            self.log_probs.append(log_prob)

        return QueryPlan(
            query=query,
            backends=selected_backends,
            query_embedding=embedding.squeeze(0).cpu().numpy().tolist(),
        )

    def reward(self, r: float) -> None:
        """Record a reward for the last decision."""
        self.rewards.append(r)

    def train_step(self) -> float:
        """
        Perform one REINFORCE update.

        Returns:
            Loss value
        """
        if not self.rewards:
            return 0.0

        # Compute returns (simple discounted sum)
        returns = []
        G = 0
        gamma = self.config.rl.gamma
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear trajectory
        self.log_probs = []
        self.rewards = []

        return loss.item()

    def _load_saved_registry(self) -> SchemaRegistry:
        """
        Load registry from saved pickle file if it exists.

        Returns:
            SchemaRegistry instance (empty if file not found)
        """
        registry_path = Path(self.config.data.processed_dir) / "registry.pkl"
        if registry_path.exists():
            try:
                import pickle
                with open(registry_path, "rb") as f:
                    registry = pickle.load(f)
                print(f"Loaded registry from {registry_path} ({len(registry.list_datasets())} datasets)")
                return registry
            except Exception as e:
                print(f"Warning: failed to load registry from {registry_path}: {e}", file=sys.stderr)
        # Return empty registry
        return SchemaRegistry()

    def _set_seeds(self, seed: int = 42) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make PyTorch deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_queries(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load queries from JSONL file.

        Args:
            data_path: Path to JSONL file

        Returns:
            List of query dicts with keys: query, relevant_doc_ids, expected_backends
        """
        queries = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    q = json.loads(line)
                    # Convert expected_backends to StorageBackend enum
                    q['expected_backends'] = [
                        StorageBackend(b) if isinstance(b, str) else b
                        for b in q.get('expected_backends', [])
                    ]
                    queries.append(q)
        return queries

    def _compute_f1(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[Any]
    ) -> float:
        """
        Compute F1 score between retrieved and relevant document IDs.

        Args:
            retrieved_ids: List of retrieved document IDs (as strings)
            relevant_ids: List of ground truth relevant document IDs

        Returns:
            F1 score (0-1), or 0 if no retrieved and no relevant
        """
        # Convert retrieved to set of strings for comparison
        retrieved_set = set(str(x) for x in retrieved_ids)
        relevant_set = set(str(x) for x in relevant_ids)

        if not retrieved_set and not relevant_set:
            return 1.0  # Perfect if both empty (edge case)

        if not retrieved_set or not relevant_set:
            return 0.0

        intersection = retrieved_set.intersection(relevant_set)
        precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(intersection) / len(relevant_set) if relevant_set else 0.0

        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _execute_retrieval(
        self,
        query: str,
        backends: List[StorageBackend],
        k: int = 10
    ) -> List[RetrievalResult]:
        """
        Execute retrieval on selected backends and collect results.

        Args:
            query: Query string
            backends: List of storage backends to query
            k: Number of results per backend

        Returns:
            Combined list of retrieval results (deduplicated by document_id)
        """
        # Load config for storage backends
        config = self.config  # Already loaded in __init__

        # Map backend to BackendConfig
        backend_configs = {
            StorageBackend.VECTOR: BackendConfig(
                backend_type=StorageBackend.VECTOR,
                connection_string=config.storage.vector.persist_directory,
                collection_name=config.storage.vector.collection_name,
                extra_params={"embedding_model": config.storage.vector.embedding_model},
            ),
            StorageBackend.GRAPH: BackendConfig(
                backend_type=StorageBackend.GRAPH,
                connection_string=config.storage.graph.uri,
                username=config.storage.graph.username,
                password=config.storage.graph.password,
                database=config.storage.graph.database,
            ),
            StorageBackend.SQL: BackendConfig(
                backend_type=StorageBackend.SQL,
                connection_string=config.storage.sql.database,
                extra_params={"echo": config.storage.sql.echo},
            ),
        }

        all_results = []
        seen_ids = set()

        for backend in backends:
            if backend not in backend_configs:
                print(f"Warning: no config for backend {backend}, skipping", file=sys.stderr)
                continue

            try:
                backend_config = backend_configs[backend]
                retriever = create_retriever_for_backend(backend, backend_config)
                results = retriever.search(query, k=k)

                # Deduplicate by document_id
                for r in results:
                    if r.document_id not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r.document_id)

                retriever.close()
            except Exception as e:
                print(f"Warning: retrieval failed for backend {backend}: {e}", file=sys.stderr)
                continue

        return all_results

    def train(
        self,
        train_data_path: Optional[str] = None,
        val_data_path: Optional[str] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        reward_scale: Optional[float] = None,
        seed: int = 42,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        save_dir: Optional[str] = None,
        k: int = 10,
    ) -> None:
        """
        Train the router policy using REINFORCE algorithm.

        Args:
            train_data_path: Path to training JSONL file (default from config)
            val_data_path: Path to validation JSONL file (default from config)
            epochs: Number of training epochs (overrides config)
            learning_rate: Learning rate (overrides config)
            reward_scale: Scale factor for rewards (overrides config)
            seed: Random seed for reproducibility
            wandb_project: WandB project name (overrides config)
            wandb_entity: WandB entity (overrides config)
            save_dir: Directory to save checkpoints (default from config)
            k: Number of documents to retrieve per backend
        """
        # Set seeds for reproducibility
        self._set_seeds(seed)

        # Override config values if provided
        if epochs is not None:
            self.config.rl.num_epochs = epochs
        if learning_rate is not None:
            self.config.rl.learning_rate = learning_rate
            # Update optimizer with new LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        if reward_scale is not None:
            self.config.rl.reward_scale = reward_scale
        if wandb_project is not None:
            self.config.logging.wandb_project = wandb_project
        if wandb_entity is not None:
            self.config.logging.wandb_entity = wandb_entity

        # Use config defaults if not specified
        train_data_path = train_data_path or self.config.data.queries_train
        val_data_path = val_data_path or self.config.data.queries_val
        save_dir = save_dir or self.config.logging.log_dir

        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print(f"Loading training data from {train_data_path}")
        train_queries = self._load_queries(train_data_path)
        print(f"Loaded {len(train_queries)} training queries")

        val_queries = []
        if val_data_path and Path(val_data_path).exists():
            print(f"Loading validation data from {val_data_path}")
            val_queries = self._load_queries(val_data_path)
            print(f"Loaded {len(val_queries)} validation queries")
        else:
            print("No validation data provided, skipping validation")

        # Initialize WandB if configured
        wandb = None
        if self.config.logging.wandb_project:
            try:
                import wandb as wb
                wandb = wb
                wandb.init(
                    project=self.config.logging.wandb_project,
                    entity=self.config.logging.wandb_entity,
                    config={
                        "epochs": self.config.rl.num_epochs,
                        "learning_rate": self.config.rl.learning_rate,
                        "gamma": self.config.rl.gamma,
                        "reward_scale": self.config.rl.reward_scale,
                        "entropy_coeff": self.config.rl.entropy_coeff,
                        "seed": seed,
                        "train_size": len(train_queries),
                        "val_size": len(val_queries),
                    }
                )
                print(f"WandB logging enabled: {self.config.logging.wandb_project}")
            except ImportError:
                print("WandB not installed, skipping logging", file=sys.stderr)
                wandb = None

        # Training loop
        best_val_f1 = 0.0
        for epoch in range(self.config.rl.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.rl.num_epochs}")
            print(f"{'='*60}")

            # Training phase
            self.policy.train()
            epoch_rewards = []
            epoch_losses = []
            num_episodes = 0

            for idx, query_data in enumerate(train_queries, 1):
                query = query_data['query']
                relevant_ids = query_data['relevant_doc_ids']
                expected_backends = query_data.get('expected_backends', [])

                # Get router decision (stores log_prob internally)
                plan = self.decide(query, eval_mode=False)

                # Execute retrieval on chosen backends
                retrieved_results = self._execute_retrieval(query, plan.backends, k=k)
                retrieved_ids = [r.document_id for r in retrieved_results]

                # Compute F1 reward
                f1 = self._compute_f1(retrieved_ids, relevant_ids)

                # Optional bonus for matching expected backend (if provided)
                bonus = 0.0
                if expected_backends:
                    # Check if chosen backends match expected backends (order-independent)
                    chosen_set = set(plan.backends)
                    expected_set = set(expected_backends)
                    if chosen_set == expected_set:
                        bonus = 0.1  # Small bonus for correct backend selection

                # Scale reward
                total_reward = (f1 * self.config.rl.reward_scale) + bonus
                self.reward(total_reward)

                epoch_rewards.append(f1)  # Track raw F1 for logging
                num_episodes += 1

                # Log progress
                if idx % 10 == 0 or idx == len(train_queries):
                    avg_f1 = np.mean(epoch_rewards) if epoch_rewards else 0.0
                    print(f"  [{idx}/{len(train_queries)}] avg_f1={avg_f1:.4f}")

            # Perform policy gradient update
            loss = self.train_step()
            epoch_losses.append(loss)

            # Log epoch metrics
            avg_train_f1 = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train F1: {avg_train_f1:.4f}")
            print(f"  Train Loss: {avg_train_loss:.4f}")

            # Validation phase
            val_f1 = 0.0
            if val_queries:
                self.policy.eval()
                val_rewards = []
                with torch.no_grad():
                    for val_data in val_queries:
                        query = val_data['query']
                        relevant_ids = val_data['relevant_doc_ids']

                        # Get decision without exploration
                        plan = self.decide(query, eval_mode=True)

                        # Execute retrieval
                        retrieved_results = self._execute_retrieval(query, plan.backends, k=k)
                        retrieved_ids = [r.document_id for r in retrieved_results]

                        # Compute F1
                        f1 = self._compute_f1(retrieved_ids, relevant_ids)
                        val_rewards.append(f1)

                val_f1 = float(np.mean(val_rewards)) if val_rewards else 0.0
                print(f"  Val F1: {val_f1:.4f}")

                # Save best checkpoint
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_path = save_dir / "router_best.pt"
                    self.save(best_path)
                    print(f"  New best model saved to {best_path} (val_f1={val_f1:.4f})")

            # Save checkpoint for this epoch
            epoch_path = save_dir / f"router_ep{epoch+1}.pt"
            self.save(epoch_path)
            print(f"  Checkpoint saved to {epoch_path}")

            # Log to WandB
            if wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_f1": avg_train_f1,
                    "train_loss": avg_train_loss,
                    "val_f1": val_f1,
                })

        print(f"\nTraining complete! Best validation F1: {best_val_f1:.4f}")
        if wandb:
            wandb.finish()

    def evaluate(
        self,
        test_data_path: Optional[str] = None,
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate the router on test data.

        Args:
            test_data_path: Path to test JSONL file
            k: Number of documents to retrieve per backend

        Returns:
            Dictionary of metrics (f1, precision, recall)
        """
        test_data_path = test_data_path or self.config.data.queries_test
        if not Path(test_data_path).exists():
            raise FileNotFoundError(f"Test data not found: {test_data_path}")

        test_queries = self._load_queries(test_data_path)
        print(f"Evaluating on {len(test_queries)} test queries")

        self.policy.eval()
        all_f1 = []
        all_precision = []
        all_recall = []

        with torch.no_grad():
            for test_data in test_queries:
                query = test_data['query']
                relevant_ids = test_data['relevant_doc_ids']

                plan = self.decide(query, eval_mode=True)
                retrieved_results = self._execute_retrieval(query, plan.backends, k=k)
                retrieved_ids = [r.document_id for r in retrieved_results]

                # Compute F1, precision, recall
                retrieved_set = set(str(x) for x in retrieved_ids)
                relevant_set = set(str(x) for x in relevant_ids)

                if not retrieved_set and not relevant_set:
                    f1 = 1.0
                    precision = 1.0
                    recall = 1.0
                elif not retrieved_set or not relevant_set:
                    f1 = 0.0
                    precision = 0.0 if not retrieved_set else len(retrieved_set.intersection(relevant_set)) / len(retrieved_set)
                    recall = 0.0 if not relevant_set else len(retrieved_set.intersection(relevant_set)) / len(relevant_set)
                else:
                    intersection = retrieved_set.intersection(relevant_set)
                    precision = len(intersection) / len(retrieved_set)
                    recall = len(intersection) / len(relevant_set)
                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)

                all_f1.append(f1)
                all_precision.append(precision)
                all_recall.append(recall)

        metrics = {
            "f1": float(np.mean(all_f1)),
            "precision": float(np.mean(all_precision)),
            "recall": float(np.mean(all_recall)),
            "num_queries": len(test_queries),
        }
        print(f"Test Results:")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        return metrics

    def save(self, path: str | Path) -> None:
        """Save policy checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str | Path) -> None:
        """Load policy checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
