# Semantic Orchestrator Architecture

## System Overview

The Semantic Orchestrator is a research-grade multi-agent system for semantic data querying across heterogeneous storage backends. It demonstrates intelligent routing of natural language queries to the most appropriate storage backend(s) using reinforcement learning, then synthesizes results into coherent answers.

### Key Use Cases

- **Multi-backend semantic search**: Query across vector (semantic similarity), graph (relationships), and SQL (structured) databases simultaneously
- **Learned query routing**: RL-based agent that learns which backend(s) are optimal for different query types
- **Result fusion**: Deduplication, reranking, and synthesis of results from multiple sources
- **Schema-aware ingestion**: Automatic schema inference from CSV data with semantic type detection

### High-Level Architecture

```mermaid
graph TB
    subgraph "Ingestion Pipeline"
        CSV[CSV/JSON Data] --> Loader[CSVLoader]
        Loader --> Schema[Schema Inference]
        Schema --> Records[DataRecords]
        Records --> VectorStore[ChromaDB]
        Records --> GraphStore[Neo4j]
        Records --> SQLite[SQLite]
    end

    subgraph "Query Pipeline"
        Query[User Query] --> Router[RL Router]
        Router --> Plan[QueryPlan<br/>Backend Selection]
        Plan --> Retriever1[VectorRetriever]
        Plan --> Retriever2[GraphRetriever]
        Plan --> Retriever3[SQLRetriever]
        Retriever1 --> Results[RetrievalResults]
        Retriever2 --> Results
        Retriever3 --> Results
        Results --> Synthesis[SynthesisAgent]
        Synthesis --> Answer[Final Answer]
    end

    subgraph "Coordination"
        Registry[SchemaRegistry]
        Config[config.yaml]
    end

    Router -.-> Registry
    Retrievers -.-> Registry
    Config -.-> all Components

    style Router fill:#e1f5fe
    style Synthesis fill:#f3e5f5
    style Registry fill:#e8f5e8
```

## Core Components

### 1. Ingestion & Schema Discovery

**Location**: `src/semantic_orchestrator/ingestion.py`

The ingestion module handles data loading and schema discovery from CSV files.

#### CSVLoader

```python
class CSVLoader:
    def load(path: str) -> pd.DataFrame
    def infer_schema(df: DataFrame, dataset_name: str) -> DatasetSchema
    def to_records(df: DataFrame, dataset_name: str) -> List[DataRecord]
```

- **`load`**: Reads CSV with configurable encoding/delimiter using pandas
- **`infer_schema`**: Automatically detects field types and **semantic types**:
  - `field_type`: Pandas dtype (e.g., `"int64"`, `"float64"`, `"object"`)
  - `semantic_type`: Semantic classification:
    - `identifier`: IDs, keys, codes
    - `measure`: Numeric metrics (price, quantity)
    - `dimension`: Categorical fields (status, region, type)
    - `text`: Free-form text content (descriptions, reviews)
    - `temporal`: Dates and times
- **`to_records`**: Converts DataFrame rows to `DataRecord` objects with enriched metadata

#### Schema Inference Heuristics

The `_guess_semantic_type` method uses:
- Column name keywords (`"date"`, `"text"`, `"description"`, `"id"`, `"category"`)
- Data types (numeric, datetime, string)
- Cardinality analysis (unique ratio < 10% suggests dimension)
- Manual override capability via field mapping

### 2. Storage Backends

The system supports three heterogeneous storage backends, each optimized for different query patterns.

#### Vector Store (ChromaDB)

**Location**: `src/semantic_orchestrator/storage/vector_store.py`

```python
class VectorStore:
    def __init__(config: BackendConfig)
    def add(records: List[DataRecord])
    def search(query: str, k: int, where: Optional[dict]) -> List[RetrievalResult]
```

- **Technology**: ChromaDB with persistent storage
- **Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2` by default)
- **Storage**: Full document content + embeddings
- **Query**: Semantic similarity search via vector distance
- **Use case**: "Find documents about X" / natural language search

**Key features**:
- Automatic embedding generation on ingest
- L2 distance converted to similarity score: `score = 1 / (1 + distance)`
- Metadata filtering support via Chroma's `where` clause

#### Graph Store (Neo4j)

**Location**: `src/semantic_orchestrator/storage/graph_store.py`

```python
class GraphStore:
    def __init__(config: BackendConfig)
    def add(records: List[DataRecord])
    def search(query: str, k: int, where: Optional[Dict]) -> List[RetrievalResult]
```

- **Technology**: Neo4j with Bolt protocol
- **Schema**: All documents stored as `:Document` nodes with properties
- **Current capability**: Metadata property filtering (full-text search planned)
- **Use case**: "Find customers in region X who bought product Y" / relationship queries

**Cypher query pattern**:
```cypher
MATCH (n:Document)
WHERE n.dataset = 'sales' AND n.region = 'West'
RETURN n.id, n.content, n
LIMIT $k
```

#### SQL Store (SQLite)

**Location**: `src/semantic_orchestrator/storage/sql_store.py`

```python
class SQLStore:
    def __init__(config: BackendConfig)
    def add(records: List[DataRecord])
    def search(query: str, k: int, where: Optional[Dict]) -> List[RetrievalResult]
```

- **Technology**: SQLite via SQLAlchemy
- **Schema**: Single `documents` table with `id`, `content`, `metadata` (JSON)
- **Search**: Basic `LIKE` full-text + metadata filtering
- **Use case**: Structured queries with exact matches, range filters

#### When to Use Each Backend

| Backend | Best For | Strengths | Limitations |
|---------|----------|-----------|-------------|
| **Vector** | Semantic search, natural language queries | Captures meaning, fuzzy matching | No exact filtering, slower writes |
| **Graph** | Relationship traversal, connected data | Path queries, network analysis | Simple search currently |
| **SQL** | Exact matches, structured filters | Fast lookups, transactional | No semantic understanding |

### 3. Schema Registry

**Location**: `src/semantic_orchestrator/registry.py`

The `SchemaRegistry` is the central coordination point that tracks:
- Which datasets exist (`DatasetSchema` with fields and semantic types)
- Which backend(s) each dataset is stored on

```python
class SchemaRegistry:
    def register_dataset(schema: DatasetSchema, backends: List[StorageBackend])
    def get_dataset_schema(name: str) -> Optional[DatasetSchema]
    def get_backends_for_dataset(name: str) -> List[StorageBackend]
    def is_available_on_backend(dataset_name: str, backend: StorageBackend) -> bool
```

**Critical role**: The router consults the registry to:
- Know what datasets are available
- Mask invalid actions (backends with no registered data)
- Make informed routing decisions based on semantic types

**Persistence**: Saved as pickle in `data/processed/registry.pkl` after ingestion.

### 4. Retrievers

**Location**: `src/semantic_orchestrator/retrieval/retrievers.py`

Retrievers are backend-specific query executors that follow the `BaseRetriever` interface:

```python
class BaseRetriever(ABC):
    def __init__(backend: StorageBackend, config: BackendConfig)
    @abstractmethod
    def search(query: str, k: int, filters: Optional[Dict]) -> List[RetrievalResult]
```

**Available retrievers**:
- `VectorRetriever` → `VectorStore`
- `GraphRetriever` → `GraphStore`
- `SQLRetriever` → `SQLStore`

**Factory function**:
```python
def create_retriever_for_backend(backend: StorageBackend, config: BackendConfig) -> BaseRetriever
```

All retrievers return standardized `RetrievalResult` objects:
```python
@dataclass
class RetrievalResult:
    document_id: str
    content: str
    score: float  # 0-1 relevance
    source_backend: StorageBackend
    metadata: Dict[str, Any]
```

### 5. Router Agent (RL-Based)

**Location**: `src/semantic_orchestrator/router.py`

The router is the system's "brain" — a reinforcement learning agent that learns to select the optimal backend(s) for a given query.

#### Action Space

7 discrete actions (all combinations of backends):

```python
ACTIONS = [
    [StorageBackend.VECTOR],               # Vector only
    [StorageBackend.GRAPH],               # Graph only
    [StorageBackend.SQL],                 # SQL only
    [StorageBackend.VECTOR, StorageBackend.GRAPH],
    [StorageBackend.VECTOR, StorageBackend.SQL],
    [StorageBackend.GRAPH, StorageBackend.SQL],
    [StorageBackend.ALL],                 # All three
]
N_ACTIONS = 7
```

#### Policy Network

```python
class RouterPolicy(nn.Module):
    def __init__(embed_dim=384, hidden_dim=128, n_actions=7)
    def forward(x: torch.Tensor) -> torch.Tensor  # logits
```

Architecture: 3-layer MLP
- Input: Query embedding (384-dim from `all-MiniLM-L6-v2`)
- Hidden: 128 units with ReLU
- Output: 7 action logits

#### Decision Process

```python
def decide(query: str, available_datasets: Optional[List[str]], eval_mode: bool) -> QueryPlan
```

1. Embed query using SentenceTransformer
2. Pass through policy network to get action logits
3. **Mask invalid actions**: Backends with no registered datasets are disabled
4. Sample action (training) or take argmax (evaluation)
5. Store `log_prob` for policy gradient training
6. Return `QueryPlan` with selected backends

#### REINFORCE Algorithm

```python
def train_step() -> float:
    # Compute discounted returns: G_t = r_t + γ * G_{t+1}
    returns = compute_discounted_returns(self.rewards, gamma)

    # Normalize returns (reduce variance)
    returns = (returns - mean) / (std + eps)

    # Policy loss: -log_prob(a|s) * G
    policy_loss = -log_prob * returns

    # Gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Training loop** (external):
1. Router makes decisions for a batch of queries
2. System executes queries, retrieves results, synthesizes answer
3. Reward function evaluates answer quality (see Evaluation)
4. `router.reward(r)` called for each decision
5. `router.train_step()` updates policy

#### Why REINFORCE?

- **Simplicity**: Single-query episodic learning, easy to implement
- **Continual learning**: Can learn from online interactions
- **Stochastic policy**: Exploration via action sampling
- **Baseline comparison**: Future work could add value network baseline

### 6. Synthesis Agent

**Location**: `src/semantic_orchestrator/synthesis.py`

The `SynthesisAgent` processes raw retrieval results into a final answer.

#### Pipeline

```python
def process(query: str, results: List[RetrievalResult], deduplicate=True, rerank=True, top_k=10) -> str
```

1. **Deduplication** (`deduplicate`):
   - Remove duplicate documents by `document_id`
   - Keep highest-scoring version if duplicates exist

2. **Reranking** (`rerank`):
   - Simple BM25-like keyword overlap scoring
   - Combined score = `vector_score + keyword_overlap`
   - Sorts by combined score, takes top `k`

3. **Synthesis** (`synthesize`):
   - Build context string with citations: `[1] (vector): content`
   - Truncate to `max_context_length` tokens
   - Call LLM (OpenRouter) with system prompt
   - Return answer with source citations

#### LLM Configuration

- **Provider**: OpenRouter (access to multiple models)
- **Model**: `google/gemma-7b-it` (configurable)
- **Temperature**: 0.1 (low for factual accuracy)
- **Prompt**: "Answer based on provided context, cite sources [1], [2]..."

### 7. Query Orchestrator Pipeline

While not yet a single class, the query execution flow is:

```python
# Pseudocode for full pipeline
def execute_query(query: str, registry: SchemaRegistry, config: Config) -> str:
    # 1. Router decides which backends to query
    plan = router.decide(query, registry.list_datasets(), eval_mode=True)

    # 2. Create retrievers for selected backends
    retrievers = [
        create_retriever_for_backend(backend, config_for_backend)
        for backend in plan.backends
    ]

    # 3. Execute searches in parallel (conceptually)
    all_results = []
    for retriever in retrievers:
        results = retriever.search(query, k=10)
        all_results.extend(results)

    # 4. Synthesize final answer
    answer = synthesis_agent.process(query, all_results)

    return answer
```

## Data Flow

### 1. Ingestion Flow

```
CSV File
   |
   v
CSVLoader.load() --> DataFrame
   |
   v
CSVLoader.infer_schema() --> DatasetSchema (with semantic types)
   |
   v
CSVLoader.to_records() --> List[DataRecord]
   |
   +------------------+
   |                  |
   v                  v
VectorStore.add()  GraphStore.add()  SQLStore.add()
   |                  |                  |
   +------------------+------------------+
                      |
                      v
            SchemaRegistry.register_dataset()
                      |
                      v
            data/processed/registry.pkl
```

**Backend Assignment Logic** (`scripts/load_data.py:assign_backends`):

| Semantic Type | Assigned Backend |
|---------------|------------------|
| `text` | Vector |
| `identifier` | Graph, SQL |
| `dimension` | Graph |
| `measure` | SQL |
| `temporal` | SQL |

All records are stored in all assigned backends with appropriate metadata.

### 2. Query Flow

```
User Query
   |
   v
+-----------------------------+
| RouterAgent.decide()        |
| - Embed query               |
| - Policy network forward   |
| - Mask invalid actions     |
| - Sample/select action     |
+-----------------------------+
   |
   v
QueryPlan (selected backends)
   |
   +-----------------------------+
   |                             |
   v                             v
[VectorRetriever]         [GraphRetriever]     [SQLRetriever]
   |                             |                  |
   v                             v                  v
VectorStore.search()      GraphStore.search()  SQLStore.search()
   |                             |                  |
   +-------------+---------------+
                 |
                 v
        Combined RetrievalResults
                 |
                 v
        SynthesisAgent.process()
                 |
       +---------+---------+
       |         |         |
       v         v         v
    Dedup    Rerank   LLM Call
       |         |         |
       +---------+---------+
                 |
                 v
            Final Answer
```

### 3. Training Flow

```
For each training query:
   |
   v
router.decide(query) --> QueryPlan
   |
   v
Execute retrieval from selected backends
   |
   v
synthesize(query, results) --> answer
   |
   v
reward_fn(query, answer, ground_truth) --> reward
   |
   v
router.reward(reward)  (store in replay)
   |
   v
After N episodes: router.train_step()
   |
   v
Policy gradient update
```

## Design Decisions

### Why Three Heterogeneous Backends?

**Problem**: Different data types and query patterns require different storage engines. A single backend cannot optimize for both semantic similarity and exact structured filtering.

**Solution**: Store data redundantly in multiple backends, let RL learn which is best for each query type.

**Trade-offs**:
- ✅ Flexibility: Each backend excels at its specialty
- ✅ Robustness: Fallback if one backend is unavailable
- ❌ Storage cost: 3x storage (acceptable for research)
- ❌ Ingestion complexity: Multiple write paths

### Why REINFORCE over PPO or DQN?

**Considered alternatives**:
- **DQN**: Discrete actions suitable, but requires replay buffer and target network → more complex
- **PPO**: More stable, but requires value network and more computation
- **A2C**: Simpler than PPO but still needs value function

**Chosen: REINFORCE (Vanilla Policy Gradient)**

**Pros**:
- Simplicity: No value network, just policy
- Natural for episodic tasks (one decision per query)
- Easy to implement with PyTorch autograd
- Suitable for small action space (7 actions)

**Cons**:
- High variance (mitigated by batch training, reward normalization)
- No built-in exploration (achieved via softmax sampling)

**Future upgrade path**: The `RouterPolicy` class can be extended with a value head for PPO without breaking API.

### Why Separate Concerns This Way?

**Design principles**:
1. **Single responsibility**: Each class has one clear purpose
2. **Interface segregation**: `BaseRetriever` abstract class, not monolithic
3. **Dependency injection**: `SchemaRegistry` passed to router, config passed universally
4. **Lazy initialization**: Embedding models loaded on first use to reduce startup cost

**Benefits**:
- Easy to test components in isolation
- Simple to add new backends (implement storage interface + retriever)
- Router doesn't need to know storage details
- Configuration centralized in Pydantic models

### Why Use SchemaRegistry?

**Problem**: Router needs to know what data exists and where without hardcoding.

**Solution**: Central registry prevents:
- Routing to empty backends (masking)
- Confusion about dataset availability
- Scattered metadata about storage assignments

**Alternative considered**: Query each backend directly to check data presence → too slow.

### Why LLM Synthesis instead of simple concatenation?

**Goal**: Natural, coherent answers that integrate multiple sources.

**Options**:
1. Simple concatenation → readable but not synthesized
2. Extract-then-synthesize (RAG style) → better but requires LLM
3. LLM-only with all results in context → chosen approach

**Rationale**: LLM can:
- Resolve contradictions between sources
- Summarize redundant information
- Generate citations naturally
- Handle cases where no source contains answer

**Cost**: One LLM call per query (mitigated by caching, cheaper models)

## Extension Points

### 1. Add a New Storage Backend

**Step 1**: Implement storage interface in `src/semantic_orchestrator/storage/`:

```python
# new_store.py
from .base import BaseStore  # or define your own
from ..types import DataRecord, BackendConfig, RetrievalResult, StorageBackend

class NewStore:
    def __init__(config: BackendConfig):
        # Validate config.backend_type
        assert config.backend_type == StorageBackend.NEW
        # Initialize connection

    def add(records: List[DataRecord]) -> None:
        # Serialize records to your backend

    def search(query: str, k: int, where: Optional[Dict]) -> List[RetrievalResult]:
        # Query backend, return RetrievalResult objects with score in [0,1]
        return [...]
```

**Step 2**: Add retriever in `src/semantic_orchestrator/retrieval/retrievers.py`:

```python
class NewRetriever(BaseRetriever):
    def _create_store(self):
        return NewStore(self.config)

    def search(self, query, k=10, filters=None):
        return self._store.search(query, k=k, where=filters)
```

**Step 3**: Update factory function:

```python
def create_retriever_for_backend(backend, config):
    if backend == StorageBackend.NEW:
        return NewRetriever(backend, config)
    # ... existing cases
```

**Step 4**: Add to `StorageBackend` enum in `types.py`:

```python
class StorageBackend(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    SQL = "sql"
    NEW = "new"
    # ...
```

**Step 5**: Update `ACTIONS` in `router.py` to include new combinations if desired.

### 2. Use a Different Embedding Model

**Option A**: Change global config (`config.yaml`):

```yaml
storage:
  vector:
    embedding_model: "sentence-transformers/all-mpnet-base-v2"
```

**Option B**: Override programmatically:

```python
from semantic_orchestrator.router import RouterAgent

agent = RouterAgent()
agent.embedder = SentenceTransformer("custom-model")
# or
config.storage.vector.embedding_model = "custom-model"
agent = RouterAgent(config_path="path/to/config.yaml")
```

**Note**: The embedding model must be compatible with SentenceTransformer API and produce the expected dimension (adjust `RouterPolicy.embed_dim` if different).

### 3. Replace the Router Algorithm

**Step 1**: Define interface to match `RouterAgent`:

```python
class BaseRouter(ABC):
    @abstractmethod
    def decide(query: str, available_datasets: List[str], eval_mode: bool) -> QueryPlan:
        pass

    @abstractmethod
    def reward(reward: float) -> None:
        pass

    @abstractmethod
    def train_step() -> float:
        pass

    @abstractmethod
    def save(path: str) -> None: ...

    @abstractmethod
    def load(path: str) -> None: ...
```

**Step 2**: Implement your algorithm:

```python
class MyCustomRouter(BaseRouter):
    def __init__(config, registry):
        self.config = config
        self.registry = registry
        # Your policy network or algorithm

    def decide(query, available_datasets, eval_mode):
        # Your decision logic
        return QueryPlan(query=query, backends=[StorageBackend.VECTOR])
```

**Step 3**: Use in place of `RouterAgent`:

```python
router = MyCustomRouter(config, registry)
# Rest of pipeline unchanged
```

**Considerations**:
- Action space definition should match `ACTIONS` or define your own
- Must return `QueryPlan` with at least one backend
- Mask invalid backends using `registry.is_available_on_backend()`

### 4. Add New Retrieval Strategies

**Per-backend strategy**: Modify individual retriever's `search` method:

```python
class EnhancedVectorRetriever(VectorRetriever):
    def search(self, query, k=10, filters=None):
        # Hybrid search: vector + keyword boost
        vector_results = self._store.search(query, k=k*2, where=filters)
        # Apply additional ranking
        return rerank_with_keywords(query, vector_results)[:k]
```

**Cross-backend fusion**: Modify synthesis or add intermediate fusion layer:

```python
def weighted_fusion(results: List[RetrievalResult], weights: Dict[StorageBackend, float]):
    # Normalize scores by backend, apply weights
    for r in results:
        r.score *= weights.get(r.source_backend, 1.0)
    return sorted(results, key=lambda r: r.score, reverse=True)
```

### 5. Integrate with Agent Lightning's Trainer/Runner

The codebase uses `agentlightning` package (see `pyproject.toml`). Integration:

```python
from agentlightning import Trainer, Runner
from semantic_orchestrator.router import RouterAgent

class SemanticOrchestratorLightning(RouterAgent):
    """Router that syncs with Agent Lightning."""

    def on_train_batch_end(self, batch, batch_idx):
        # Called by Trainer after batch
        self.train_step()

    def on_validation_epoch_end(self):
        # Log metrics to wandb
        return {"val_reward": avg_reward}

# Configuration
trainer = Trainer(
    max_epochs=config.rl.num_epochs,
    logger=wandb_logger if config.logging.wandb_project else None,
)

router = SemanticOrchestratorLightning(config)
trainer.fit(router, train_dataloader, val_dataloader)
```

**Key hooks**:
- `training_step()`: Make decisions, compute rewards, return loss
- `validation_step()`: Evaluate without exploration
- `configure_optimizers()`: Return optimizer (already in RouterAgent)

## Evaluation Methodology

### 1. Router Quality

**Metrics**:
- **Accuracy**: % of queries where selected backend(s) return relevant results
- **Efficiency**: Average number of backends selected (fewer = cheaper)
- **Reward**: Moving average of episode rewards during training

**Evaluation procedure**:
```python
test_queries = load_test_dataset()  # (query, ground_truth) pairs
rewards = []

for query, ground_truth in test_queries:
    plan = router.decide(query, eval_mode=True)
    results = execute_retrieval(plan.backends, query)
    answer = synthesis.process(query, results)

    reward = compute_reward(answer, ground_truth)
    rewards.append(reward)

avg_reward = np.mean(rewards)
```

### 2. Retrieval Performance

**Standard IR metrics** (computed per backend with ground truth relevant docs):

- **Precision@k**: `(# relevant in top-k) / k`
- **Recall@k**: `(# relevant in top-k) / (total relevant)`
- **F1@k**: Harmonic mean of P@k and R@k
- **MRR** (Mean Reciprocal Rank): `mean(1 / rank_of_first_relevant)`

**Tooling**: Use `sklearn.metrics` or `trec_eval`.

### 3. Synthesis Accuracy

**LLM-as-a-Judge** (automated):

```python
judge_prompt = f"""
Compare answer with ground truth. Score 1-5:
1: Completely wrong
3: Partially correct
5: Fully correct, same information

Question: {query}
Answer: {answer}
Ground Truth: {ground_truth}

Score:
"""

score = llm_call(judge_prompt)
```

**Human evaluation** (research quality):
- 100 random test queries
- 3 human annotators
- Metrics: correctness, completeness, hallucination rate
- Inter-annotator agreement (Cohen's κ)

### 4. End-to-End Pipeline

**Composite metrics**:
- **Answer Quality**: LLM-as-judge score
- **Latency**: Time from query to answer
- **Cost**: API calls × token count (USD)
- **Backend Utilization**: Distribution of selected backends

**Ablation studies**:
- Router vs. random backend selection
- With/without deduplication
- With/without reranking
- Single vs. multi-backend

## Configuration Reference

**File**: `config.yaml` (or custom path)

### Storage Configuration

```yaml
storage:
  vector:
    collection_name: "semantic_docs"      # Chroma collection name
    embedding_model: "all-MiniLM-L6-v2"  # SentenceTransformer model
    persist_directory: "./data/chroma"   # Chroma storage path

  graph:
    uri: "bolt://localhost:7687"         # Neo4j Bolt URI
    username: "neo4j"
    password: "password"
    database: null                       # Optional database name

  sql:
    database: "./data/sqlite/semantic.db" # SQLite file path
    echo: false                          # SQLAlchemy echo (debug)
```

### Router Configuration

```yaml
router:
  model: "google/gemma-7b-it"    # OpenRouter model for synthesis
  temperature: 0.1                # LLM temperature (0-1)
  max_tokens: 100                # Max tokens in response
  top_p: 0.9                     # Nucleus sampling
```

**Note**: Router policy uses sentence-transformer model from `storage.vector.embedding_model`.

### RL Training Configuration

```yaml
rl:
  algorithm: "reinforce"         # or "ppo" (future)
  learning_rate: 1e-4            # Policy network LR
  gamma: 0.99                    # Discount factor
  batch_size: 4                  # Episodes per update
  num_epochs: 50                 # Training epochs
  reward_scale: 1.0              # Multiply rewards by this
  entropy_coeff: 0.01            # Entropy regularization
```

### Data Paths

```yaml
data:
  raw_dir: "./data/raw"                    # CSV input files
  processed_dir: "./data/processed"        # Registry, caches
  embeddings_cache: "./data/embeddings_cache"  # Future: cache embeddings
  queries_train: "./data/queries/train.jsonl"
  queries_val: "./data/queries/val.jsonl"
  queries_test: "./data/queries/test.jsonl"
```

**Query file format** (JSONL):
```json
{"query": "What are sales in Q1?", "ground_truth": "..."}
```

### Logging & Experiment Tracking

```yaml
logging:
  wandb_project: "agent-lightning-semantic"  # Weights & Biases project
  wandb_entity: null                         # Username/org (set if needed)
  log_dir: "./logs"                          # Local logs
  level: "INFO"                              # Logging level
```

### Complete Example

```yaml
storage:
  vector:
    collection_name: "docs"
    embedding_model: "all-MiniLM-L6-v2"
    persist_directory: "./chroma"
  graph:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "secret123"
  sql:
    database: "./data.db"

router:
  model: "anthropic/claude-3-haiku"
  temperature: 0.0
  max_tokens: 200

rl:
  algorithm: "reinforce"
  learning_rate: 1e-4
  gamma: 0.95
  batch_size: 8
  num_epochs: 100

data:
  raw_dir: "./data/raw"
  processed_dir: "./data/processed"
  queries_train: "./data/train.jsonl"

logging:
  wandb_project: "my-semantic-orch"
  level: "DEBUG"
```

## Summary

The Semantic Orchestrator is a modular, research-oriented system that combines:
- **Multiple heterogeneous stores** for optimal query performance
- **RL-based routing** that learns from feedback
- **Modular architecture** for easy experimentation and extension
- **Production-aware design** with config, logging, and error handling

Key file reference:
- `src/semantic_orchestrator/types.py` — Core data models
- `src/semantic_orchestrator/ingestion.py` — CSV loading & schema inference
- `src/semantic_orchestrator/registry.py` — Dataset/backend coordination
- `src/semantic_orchestrator/router.py` — RL policy network & decision logic
- `src/semantic_orchestrator/retrieval/retrievers.py` — Backend query execution
- `src/semantic_orchestrator/synthesis.py` — Result fusion & LLM answer
- `scripts/load_data.py` — End-to-end ingestion script
- `config.yaml` — Full configuration reference
