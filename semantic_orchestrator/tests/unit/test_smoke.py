"""Smoke test - verify basic project structure and imports."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def test_project_structure():
    """Test that key directories exist."""
    root = Path(__file__).parent.parent.parent
    assert (root / "src").exists(), "src directory missing"
    # data will be created lazily, don't enforce existence yet
    print("✓ Project structure OK")

def test_imports():
    """Test that core dependencies can be imported."""
    import pandas as pd
    import chromadb
    import neo4j
    from sentence_transformers import SentenceTransformer
    print("✓ Core dependencies importable")

def test_config_load():
    """Test that config can be loaded."""
    import yaml
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "storage" in config, "Missing storage config"
        assert "router" in config, "Missing router config"
        print("✓ Config loadable")
    else:
        print("⚠ Config file not found (will be created)")
