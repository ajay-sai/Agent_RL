"""Configuration management using Pydantic."""
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class VectorConfig(BaseModel):
    """Configuration for vector storage (Chroma)."""
    collection_name: str = "semantic_docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    persist_directory: str = "./data/chroma"


class GraphConfig(BaseModel):
    """Configuration for graph storage (Neo4j)."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: Optional[str] = None


class SQLConfig(BaseModel):
    """Configuration for relational storage (SQLite)."""
    database: str = "./data/sqlite/semantic.db"
    echo: bool = False


class StorageConfig(BaseModel):
    """Complete storage configuration."""
    vector: VectorConfig = Field(default_factory=VectorConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    sql: SQLConfig = Field(default_factory=SQLConfig)


class RouterConfig(BaseModel):
    """Configuration for the router agent."""
    model: str = "google/gemma-7b-it"
    temperature: float = 0.1
    max_tokens: int = 100
    top_p: float = 0.9


class RLConfig(BaseModel):
    """Reinforcement learning training configuration."""
    algorithm: str = "reinforce"
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 4
    num_epochs: int = 50
    reward_scale: float = 1.0
    entropy_coeff: float = 0.01


class DataConfig(BaseModel):
    """Data paths and locations."""
    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    embeddings_cache: str = "./data/embeddings_cache"
    queries_train: str = "./data/queries/train.jsonl"
    queries_val: str = "./data/queries/val.jsonl"
    queries_test: str = "./data/queries/test.jsonl"


class LoggingConfig(BaseModel):
    """Logging and experiment tracking."""
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_dir: str = "./logs"
    level: str = "INFO"


class Config(BaseModel):
    """Main configuration object."""
    storage: StorageConfig = Field(default_factory=StorageConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


default_config = Config()


def load_config(config_path: Optional[str] = None) -> Config:
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return Config()
