"""Tests for synthesis agent."""
from semantic_orchestrator.synthesis import SynthesisAgent
from semantic_orchestrator.types import RetrievalResult, StorageBackend

def test_deduplicate_removes_duplicates():
    """Test deduplicate removes duplicates, keeps highest score."""
    agent = SynthesisAgent()
    results = [
        RetrievalResult("doc1", "Content A", 0.8, StorageBackend.VECTOR, {}),
        RetrievalResult("doc1", "Content A", 0.9, StorageBackend.VECTOR, {}),
        RetrievalResult("doc2", "Content B", 0.7, StorageBackend.SQL, {}),
    ]
    deduped = agent.deduplicate(results)
    assert len(deduped) == 2
    doc1_result = [r for r in deduped if r.document_id == "doc1"][0]
    assert doc1_result.score == 0.9

def test_rerank():
    """Test reranking sorts by relevance."""
    agent = SynthesisAgent()
    query = "machine learning"
    results = [
        RetrievalResult("1", "Neural networks are used in deep learning", 0.6, StorageBackend.VECTOR, {}),
        RetrievalResult("2", "Machine learning algorithms", 0.5, StorageBackend.VECTOR, {}),
        RetrievalResult("3", "Random text about nothing", 0.3, StorageBackend.SQL, {}),
    ]
    reranked = agent.rerank(query, results)
    assert reranked[0].document_id == "2"

def test_process_pipeline():
    """Test full processing - requires OPENROUTER_API_KEY."""
    import os
    if not os.environ.get("OPENROUTER_API_KEY"):
        return  # Skip if no API key
    agent = SynthesisAgent()
    query = "ML"
    results = [
        RetrievalResult("doc1", "Machine learning content", 0.8, StorageBackend.VECTOR, {}),
        RetrievalResult("doc1", "Machine learning duplicate", 0.6, StorageBackend.VECTOR, {}),
        RetrievalResult("doc2", "Neural networks", 0.7, StorageBackend.SQL, {}),
    ]
    answer = agent.process(query, results, top_k=5)
    assert isinstance(answer, str) and len(answer) > 0
