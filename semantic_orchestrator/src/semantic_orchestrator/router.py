"""RL Router Agent - learns to select backend(s) for queries."""
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .config import load_config
from .registry import SchemaRegistry
from .types import QueryPlan, StorageBackend

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
            registry: SchemaRegistry instance (creates default if None)
            device: 'cpu' or 'cuda'
        """
        self.config = load_config(config_path)
        self.registry = registry or SchemaRegistry()
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
            eval_mode: If True, use argmax; else sample for exploration

        Returns:
            QueryPlan with selected backends
        """
        if available_datasets is None:
            available_datasets = self.registry.list_datasets()

        # Get query embedding
        embedding = self.embed_query(query).unsqueeze(0)  # (1, embed_dim)

        # Get action logits
        self.policy.eval()
        with torch.no_grad():
            logits = self.policy(embedding)  # (1, n_actions)

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

        # Store log probability for training
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
