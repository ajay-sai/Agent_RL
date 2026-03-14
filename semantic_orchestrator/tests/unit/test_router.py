"""Tests for RL router."""
from semantic_orchestrator.router import RouterAgent, ACTIONS, N_ACTIONS
from semantic_orchestrator.registry import SchemaRegistry
from semantic_orchestrator.types import DatasetSchema, SchemaField, StorageBackend

def test_router_initialization():
    """Test RouterAgent can be initialized."""
    agent = RouterAgent()
    assert agent.policy is not None
    assert N_ACTIONS == 7

def test_actions_coverage():
    """Test that all backend combinations are present."""
    expected = [
        [StorageBackend.VECTOR],
        [StorageBackend.GRAPH],
        [StorageBackend.SQL],
        [StorageBackend.VECTOR, StorageBackend.GRAPH],
        [StorageBackend.VECTOR, StorageBackend.SQL],
        [StorageBackend.GRAPH, StorageBackend.SQL],
        [StorageBackend.ALL],
    ]
    assert ACTIONS == expected

def test_router_decide_with_valid_datasets():
    """Test decide returns a valid QueryPlan when datasets available."""
    agent = RouterAgent()
    registry = SchemaRegistry()
    schema = DatasetSchema(
        name="test",
        fields=[SchemaField("id", "int64", "identifier", [])],
    )
    registry.register_dataset(schema, backends=[StorageBackend.VECTOR, StorageBackend.SQL])
    agent.registry = registry

    plan = agent.decide("test query", eval_mode=True)
    assert isinstance(plan, object)
    assert hasattr(plan, 'query')
    assert hasattr(plan, 'backends')
    # At least one backend should be valid
    assert len(plan.backends) > 0

def test_router_decide_masks_invalid_backends():
    """Test that router masks backends with no registered datasets."""
    agent = RouterAgent()
    registry = SchemaRegistry()
    schema = DatasetSchema(name="data", fields=[])
    registry.register_dataset(schema, backends=[StorageBackend.VECTOR])
    agent.registry = registry

    plan = agent.decide("query", eval_mode=True)
    # Should not return GRAPH or SQL only combos
    if plan.backends == [StorageBackend.GRAPH] or plan.backends == [StorageBackend.SQL]:
        assert False, f"Invalid backend selected: {plan.backends}"

def test_router_save_load():
    """Test saving and loading policy checkpoint."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "router.pt"
        agent1 = RouterAgent()
        agent1.save(path)

        agent2 = RouterAgent()
        agent2.load(path)
        # Check that weights match
        for p1, p2 in zip(agent1.policy.parameters(), agent2.policy.parameters()):
            assert torch.allclose(p1, p2)

def test_router_reward_and_train():
    """Test recording rewards and performing train step."""
    agent = RouterAgent()
    # Simulate some decisions
    agent.log_probs = [torch.tensor(-0.5, requires_grad=True), torch.tensor(-0.3, requires_grad=True)]
    agent.rewards = [1.0, 0.0]
    loss = agent.train_step()
    assert isinstance(loss, float)
    assert loss > 0  # Should have some loss
