"""Synthesis Agent - combines results from multiple backends into coherent answer."""
from typing import List, Dict, Any, Optional

from openai import OpenAI
from .config import load_config
from .types import RetrievalResult, StorageBackend


class SynthesisAgent:
    """Synthesizes answers from multiple retrieval results."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize synthesis agent.

        Args:
            config_path: Path to config.yaml
        """
        self.config = load_config(config_path)

        # Initialize OpenRouter client lazily
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy-initialize OpenRouter client."""
        if self._client is None:
            api_key = self._get_api_key()
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        return self._client

    def _get_api_key(self) -> str:
        """Get OpenRouter API key from environment or config."""
        import os
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key and hasattr(self.config, 'router'):
            api_key = getattr(self.config.router, 'openrouter_api_key', None)
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                "or add openrouter_api_key to router config."
            )
        return api_key

    def deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Remove duplicate results by document_id.

        Args:
            results: List of retrieval results (may have duplicates from multi-backend)

        Returns:
            Deduplicated list, keeping highest score per document
        """
        seen: Dict[str, RetrievalResult] = {}
        for r in results:
            if r.document_id not in seen or r.score > seen[r.document_id].score:
                seen[r.document_id] = r
        return list(seen.values())

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Rerank results by relevance to query (simple BM25-like scoring).

        Args:
            query: User query
            results: Retrieved results
            top_k: Number to return

        Returns:
            Reranked and truncated list
        """
        # Simple keyword-based reranking (for CPU efficiency)
        query_words = set(query.lower().split())

        def score(result: RetrievalResult) -> float:
            # Combine vector score with keyword overlap
            content_words = set(result.content.lower().split())
            keyword_overlap = len(query_words & content_words) / len(query_words) if query_words else 0
            return result.score + keyword_overlap

        sorted_results = sorted(results, key=score, reverse=True)
        return sorted_results[:top_k]

    def synthesize(
        self,
        query: str,
        results: List[RetrievalResult],
        max_context_length: int = 4000
    ) -> str:
        """
        Generate final answer from retrieved results.

        Args:
            query: User query
            results: Retrieved and processed results
            max_context_length: Max tokens for context

        Returns:
            Answer string with citations
        """
        # Prepare context with citations
        context_parts = []
        for i, r in enumerate(results):
            source = f"[{i+1}] ({r.source_backend.value})"
            context_parts.append(f"{source}: {r.content}")
        context = "\n\n".join(context_parts)

        # Truncate if too long (simple truncation)
        if len(context.split()) > max_context_length:
            context = " ".join(context.split()[:max_context_length])

        system_prompt = """You are a helpful assistant that answers questions based on provided context.
Synthesize information from multiple sources into a coherent, accurate answer.
If the context doesn't contain the answer, say so.
Cite sources using the numbers in brackets like [1], [2], etc."""

        user_prompt = f"""Question: {query}

Context:
{context}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.router.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.router.temperature,
                max_tokens=self.config.router.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def process(
        self,
        query: str,
        results: List[RetrievalResult],
        deduplicate: bool = True,
        rerank: bool = True,
        top_k: int = 10
    ) -> str:
        """
        Full processing pipeline: deduplicate, rerank, synthesize.

        Args:
            query: User query
            results: Raw retrieval results
            deduplicate: Whether to deduplicate
            rerank: Whether to rerank
            top_k: Max results after reranking

        Returns:
            Final answer string
        """
        if deduplicate:
            results = self.deduplicate(results)

        if rerank:
            results = self.rerank(query, results, top_k=top_k)

        return self.synthesize(query, results)
