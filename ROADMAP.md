# CapabilityMesh - Product Roadmap

**Current Version**: v1.0.0-alpha.1 (Ready for Publication)
**Last Updated**: December 9, 2025

## Vision Statement
**"pip install capabilitymesh + 5 lines = multi-agent discovery working"**

Enable AI/ML engineers to discover, connect, and coordinate agents across any framework (CrewAI, AutoGen, LangGraph, A2A) with minimal setup.

## v1.0.0-alpha.1 Status: ✅ COMPLETE

**Publication Status**: Ready for PyPI
**Test Coverage**: 139/139 tests passing (100%)
**Documentation**: Complete
**Examples**: 6 comprehensive examples

---

## Target User: AI/ML Engineers

**Persona**:
- Building production multi-agent systems
- Using multiple frameworks (experimenting or production)
- Needs agents to find and work with each other
- Values: simplicity, reliability, good DX

**Success Metric**: Developer can go from `pip install` to working cross-framework agent discovery in under 10 minutes.

---

## Core Value Propositions (All Four)

| Value | Description | Priority |
|-------|-------------|----------|
| **Discover Agents** | Find agents with specific capabilities | P0 |
| **Connect Frameworks** | CrewAI + AutoGen + LangGraph interop | P0 |
| **Standardize Capabilities** | Common capability schema | P0 |
| **A2A Compatible** | Works with A2A protocol agents | P0 |

---

## API Design (Developer Experience First)

### Quickstart (5 lines)

```python
from capabilitymesh import Mesh, capability

mesh = Mesh()  # In-memory by default

@mesh.agent(capabilities=["summarization", "translation"])
def my_agent(task: str) -> str:
    return f"Processed: {task}"

# Discover and use agents
agent = mesh.discover("summarize text")
result = agent.execute("Summarize this article...")
```

### Framework Integration (3 lines per framework)

```python
from capabilitymesh import Mesh
from crewai import Agent

mesh = Mesh()

# Wrap any framework agent
crewai_agent = Agent(role="researcher", goal="Find information")
mesh.register(crewai_agent)  # Auto-extracts capabilities

# Discover across frameworks
researcher = mesh.discover("research capabilities")
```

### Registration Patterns (All Three)

```python
# 1. Decorator pattern
@mesh.agent(name="summarizer")
def summarize(text: str) -> str: ...

# 2. Explicit registration
mesh.register(my_crewai_agent)
mesh.register(my_autogen_agent)

# 3. Auto-discovery (scan module)
mesh.auto_discover("my_agents")  # Scans my_agents/ directory
```

---

## Core Components for v1.0

### 1. Mesh (Central Hub) - NEW
The main entry point. **Async-first** API with sync wrappers.

```python
class Mesh:
    def __init__(
        self,
        storage: Storage = InMemoryStorage(),
        embedder: Embedder = None,  # Auto-selects best available
    ): ...

    # Registration (sync - immediate)
    async def register(
        self,
        agent,
        capabilities: List[str] = None  # Override auto-extracted
    ) -> AgentIdentity: ...

    def agent(self, **kwargs) -> Decorator: ...  # @mesh.agent()
    async def auto_discover(self, module_path: str) -> List[AgentIdentity]: ...

    # Discovery (async)
    async def discover(self, query: str, limit: int = 5) -> List[Agent]: ...
    async def discover_by_capability(self, capability: str) -> List[Agent]: ...
    async def list_agents(self) -> List[AgentInfo]: ...

    # Execution (async)
    async def execute(self, agent_id: str, task: str, **kwargs) -> Result: ...

    # Native access
    def get_native(self, agent_id: str) -> Any: ...  # Returns original framework agent

    # Sync wrappers (convenience)
    def discover_sync(self, query: str, limit: int = 5) -> List[Agent]: ...
    def execute_sync(self, agent_id: str, task: str) -> Result: ...
```

### 2. Storage Backend - NEW
Pluggable storage for agent registry.

```python
# Built-in options
class InMemoryStorage(Storage): ...    # Default, zero config
class RedisStorage(Storage): ...       # pip install capabilitymesh[redis]
class SQLiteStorage(Storage): ...      # Local persistent storage

# Interface for custom storage
class Storage(ABC):
    def save_agent(self, agent: AgentRecord) -> None: ...
    def get_agent(self, agent_id: str) -> AgentRecord: ...
    def search(self, query: str) -> List[AgentRecord]: ...
    def list_all(self) -> List[AgentRecord]: ...
```

### 3. Embedder (Semantic Search) - NEW
Pluggable embeddings for semantic discovery.

```python
# Auto-selection priority:
# 1. If sentence-transformers installed -> LocalEmbedder
# 2. If OPENAI_API_KEY set -> OpenAIEmbedder
# 3. Fallback -> KeywordEmbedder (no ML, just keyword matching)

class LocalEmbedder(Embedder):        # sentence-transformers
    model = "all-MiniLM-L6-v2"        # Fast, good quality

class OpenAIEmbedder(Embedder):       # OpenAI API
    model = "text-embedding-3-small"

class KeywordEmbedder(Embedder):      # No dependencies
    # TF-IDF based, works without ML
```

### 4. Trust System (Simple) - NEW
Simple trust tracking for v1.

```python
class TrustLevel(Enum):
    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4

class TrustManager:
    def set_trust(self, agent_id: str, level: TrustLevel): ...
    def get_trust(self, agent_id: str) -> TrustScore: ...
    def record_execution(self, agent_id: str, success: bool): ...

# TrustScore includes:
# - Manual trust level
# - Success rate (executions succeeded / total)
# - Total executions count
```

### 5. Negotiation (Basic) - NEW
Simple negotiation for v1.

```python
class NegotiationRequest:
    capability: str
    constraints: dict  # timeout, max_cost, etc.

class NegotiationResponse:
    accepted: bool
    terms: dict
    reason: str = None

# Simple flow:
# 1. Requester sends NegotiationRequest
# 2. Agent responds with NegotiationResponse
# 3. If accepted, execution proceeds
```

### 6. Enhanced Framework Adapters - EXISTING (enhance)
Already have CrewAI, AutoGen, LangGraph, A2A. Enhance with:
- Better auto-extraction of capabilities
- Consistent `execute()` interface
- Trust tracking integration

---

## Feature Scope

### IN SCOPE (v1.0)

| Feature | Description | Status |
|---------|-------------|--------|
| **Mesh API** | Central hub with simple API | New |
| **In-Memory Storage** | Zero-config default | New |
| **Redis Storage** | Optional persistent storage | New |
| **SQLite Storage** | Local persistent option | New |
| **Semantic Search** | Find agents by natural language | New |
| **Local Embeddings** | sentence-transformers | New |
| **OpenAI Embeddings** | Optional cloud embeddings | New |
| **Keyword Fallback** | No-ML keyword matching | New |
| **Decorator Registration** | `@mesh.agent()` | New |
| **Explicit Registration** | `mesh.register()` | New |
| **Auto-Discovery** | Scan modules for agents | New |
| **Simple Trust** | Manual levels + success rate | New |
| **Basic Negotiation** | Accept/reject with constraints | New |
| **CrewAI Integration** | Enhanced adapter | Enhance |
| **AutoGen Integration** | Enhanced adapter | Enhance |
| **LangGraph Integration** | Enhanced adapter | Enhance |
| **A2A Integration** | Enhanced adapter | Enhance |
| **Capability Schema** | Structured/Unstructured | Existing |
| **Agent Identity** | DID-based identity | Existing |

### OUT OF SCOPE (Future Releases)

| Feature | Target Version |
|---------|----------------|
| P2P Discovery (mDNS, DHT, Gossip) | v1.1 |
| Peer-to-peer trust ratings | v1.2 |
| Cryptographic attestations | v1.2 |
| Multi-round negotiation | v1.1 |
| gRPC/MQTT transport | v1.1 |
| CLI tools | v1.1 |
| Web dashboard | v2.0 |

---

## File Structure (Simplified)

```
capabilitymesh/
├── __init__.py          # Main exports: Mesh, capability, etc.
├── mesh.py              # NEW: Mesh class (main entry point)
├── core/
│   ├── identity.py      # EXISTING: AgentIdentity, AgentAddress
│   ├── types.py         # EXISTING: Enums
│   └── exceptions.py    # EXISTING: Exceptions
├── schemas/
│   └── capability.py    # EXISTING: Capability models
├── storage/             # NEW
│   ├── __init__.py
│   ├── base.py          # Storage ABC
│   ├── memory.py        # InMemoryStorage
│   ├── redis.py         # RedisStorage (optional)
│   └── sqlite.py        # SQLiteStorage
├── embeddings/          # NEW
│   ├── __init__.py
│   ├── base.py          # Embedder ABC
│   ├── local.py         # LocalEmbedder (sentence-transformers)
│   ├── openai.py        # OpenAIEmbedder
│   └── keyword.py       # KeywordEmbedder (fallback)
├── trust/               # NEW (simplified)
│   ├── __init__.py
│   └── simple.py        # SimpleTrustManager
├── negotiation/         # NEW (simplified)
│   ├── __init__.py
│   └── basic.py         # BasicNegotiator
├── discovery/           # NEW (simplified)
│   ├── __init__.py
│   └── registry.py      # AgentRegistry (in-process discovery)
└── integrations/        # EXISTING (enhanced)
    ├── base.py
    ├── crewai.py
    ├── autogen.py
    ├── langgraph.py
    └── a2a/
```

---

## Installation Options

```bash
# Minimal (keyword matching only)
pip install capabilitymesh

# With local embeddings (recommended)
pip install capabilitymesh[embeddings]

# With Redis storage
pip install capabilitymesh[redis]

# With OpenAI embeddings
pip install capabilitymesh[openai]

# With specific framework
pip install capabilitymesh[crewai]
pip install capabilitymesh[autogen]
pip install capabilitymesh[langgraph]

# Full installation
pip install capabilitymesh[all]
```

---

## Example Use Cases

### Use Case 1: Simple Agent Discovery

```python
from capabilitymesh import Mesh

mesh = Mesh()

# Register agents with capabilities
@mesh.agent(
    name="summarizer",
    capabilities=["text-summarization", "content-extraction"],
    description="Summarizes long documents into key points"
)
def summarize(text: str) -> str:
    # Your summarization logic
    return summary

@mesh.agent(
    name="translator",
    capabilities=["translation", "language-detection"],
    description="Translates text between languages"
)
def translate(text: str, target_lang: str) -> str:
    # Your translation logic
    return translated

# Discover by semantic query
agents = mesh.discover("I need to shorten this article")
# Returns: [summarizer] (semantic match)

agents = mesh.discover("convert to Spanish")
# Returns: [translator] (semantic match)
```

### Use Case 2: Cross-Framework Coordination

```python
from capabilitymesh import Mesh
from crewai import Agent as CrewAgent
from autogen import AssistantAgent

mesh = Mesh()

# Register CrewAI agent
researcher = CrewAgent(role="researcher", goal="Find information")
mesh.register(researcher)

# Register AutoGen agent
coder = AssistantAgent(name="coder", system_message="Write code")
mesh.register(coder)

# Discover by capability (framework-agnostic)
research_agent = mesh.discover("research and find information")[0]
coding_agent = mesh.discover("write python code")[0]

# Execute (unified interface)
research_result = research_agent.execute("Find best practices for API design")
code_result = coding_agent.execute(f"Implement: {research_result}")
```

### Use Case 3: Trust-Aware Discovery

```python
from capabilitymesh import Mesh, TrustLevel

mesh = Mesh()

# Register agents
mesh.register(agent1)
mesh.register(agent2)

# Set trust levels
mesh.trust.set_level(agent1.id, TrustLevel.HIGH)
mesh.trust.set_level(agent2.id, TrustLevel.LOW)

# Discover with trust filter
trusted_agents = mesh.discover(
    "summarization",
    min_trust=TrustLevel.MEDIUM
)

# Trust automatically updates based on execution success
result = mesh.execute(agent1.id, "Summarize this")
# If success -> trust score improves
# If failure -> trust score decreases
```

### Use Case 4: Basic Negotiation

```python
from capabilitymesh import Mesh

mesh = Mesh()

# Discover and negotiate
agent = mesh.discover("data processing")[0]

# Check if agent accepts our constraints
negotiation = mesh.negotiate(
    agent_id=agent.id,
    constraints={
        "max_timeout": 30,  # seconds
        "max_cost": 0.10,   # dollars
    }
)

if negotiation.accepted:
    result = agent.execute("Process this data")
else:
    print(f"Rejected: {negotiation.reason}")
```

---

## Implementation Status

### ✅ Phase 1: Core Mesh API (COMPLETE)
- [x] `Mesh` class with basic API
- [x] In-memory storage backend
- [x] Decorator registration (`@mesh.agent`)
- [x] Explicit registration (`mesh.register()`)
- [x] Basic `discover()` with keyword matching
- [x] `list_agents()` functionality

### ✅ Phase 2: Semantic Search (COMPLETE)
- [x] Embedder abstraction
- [x] KeywordEmbedder (no dependencies)
- [x] Auto-selection logic
- [x] Semantic `discover()` implementation
- [ ] LocalEmbedder (sentence-transformers) - Planned for v1.1
- [ ] OpenAIEmbedder (optional) - Planned for v1.1

### ✅ Phase 3: Storage & Trust (COMPLETE)
- [x] Storage abstraction
- [x] SQLiteStorage for persistence (with FTS5 search)
- [x] RedisStorage (optional, distributed)
- [x] SimpleTrustManager (5-level system)
- [x] Trust-aware discovery
- [x] Execution tracking with duration

### 🚀 Phase 4: Alpha Release (COMPLETE)
- [x] Comprehensive tests (139 tests, 100% passing)
- [x] Documentation (README, CHANGELOG, CONTRIBUTING)
- [x] Examples (6 comprehensive examples)
- [x] Pre-publication testing
- [x] Package cleanup
- [ ] PyPI publication (Ready to publish)

### 📋 Future Phases

#### v1.0.0-beta.1 (Next)
- [ ] Community feedback incorporation
- [ ] Real-world framework testing
- [ ] Performance optimization
- [ ] Additional storage backends
- [ ] Enhanced embeddings (LocalEmbedder, OpenAIEmbedder)

#### v1.0.0 (Stable Release)
- [ ] Production hardening
- [ ] Comprehensive documentation site
- [ ] Tutorial videos
- [ ] Case studies
- [ ] Community showcase

---

## Success Criteria

1. **Simplicity**: 5 lines to working discovery
2. **Speed**: < 100ms for in-memory discovery
3. **Flexibility**: Works with or without ML dependencies
4. **Coverage**: All 4 major frameworks supported
5. **Testing**: > 80% code coverage
6. **Docs**: Complete API reference + tutorials

---

## Design Decisions (Finalized)

1. **Async support**: **Async-first** - All methods are async (`await mesh.discover()`), with sync wrappers available for convenience.

2. **Agent execution**: **Both options** - Unified `mesh.execute(agent_id, task)` for simplicity, plus `mesh.get_native(agent_id)` to access the original framework agent.

3. **Capability extraction**: **Auto + override** - Automatically extract capabilities from framework agents, but allow explicit override with manual capabilities.

---

## Summary

v1.0 focuses on **developer experience** and **immediate value**:

| What | How |
|------|-----|
| Easy setup | `pip install` + 5 lines |
| Semantic search | Pluggable embeddings |
| Cross-framework | Unified adapter interface |
| Trust | Simple scores + success rate |
| Negotiation | Basic accept/reject |
| Storage | In-memory default, Redis/SQLite optional |

This creates a solid foundation that developers will actually use, with clear extension points for future capabilities (P2P discovery, advanced trust, multi-round negotiation).
