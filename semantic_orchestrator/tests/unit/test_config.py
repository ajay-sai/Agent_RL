"""Tests for configuration management."""
import pytest
from pathlib import Path
from semantic_orchestrator.config import load_config, Config, StorageConfig, RouterConfig, RLConfig

def test_load_config_from_yaml(tmp_path):
    """Test loading configuration from YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
storage:
  vector:
    collection_name: "test_collection"
    embedding_model: "all-MiniLM-L6-v2"
  graph:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "test"
  sql:
    database: "test.db"

router:
  model: "google/gemma-7b-it"
  temperature: 0.2

rl:
  algorithm: "reinforce"
  learning_rate: 1e-4
""")
    config = load_config(str(config_file))
    assert isinstance(config, Config)
    assert config.storage.vector.collection_name == "test_collection"
    assert config.router.model == "google/gemma-7b-it"
    assert config.rl.learning_rate == 1e-4

def test_config_defaults():
    """Test that default configuration is valid."""
    config = Config()
    assert config.storage.vector.collection_name == "semantic_docs"
    assert config.router.temperature == 0.1
    assert config.rl.gamma == 0.99

def test_config_validation_requires_vector_path():
    """Test that vector storage needs either collection or path."""
    config = Config()
    assert config.storage.vector.persist_directory is not None
