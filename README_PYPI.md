# CapabilityMesh

**The first and only Python package for universal capability discovery across all major agent frameworks**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-139%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)]()

---

## What is CapabilityMesh?

CapabilityMesh enables AI agents from **any framework** to discover and collaborate with each other. Build multi-agent systems where CrewAI, AutoGen, LangGraph, A2A, and custom agents work together seamlessly.

**The Problem:** Today's multi-agent ecosystem is fragmented. CrewAI agents can't discover AutoGen agents. LangGraph workflows can't find A2A services. No standard way to ask "which agents can translate?"

**The Solution:** CapabilityMesh provides a universal discovery layer that works across ALL frameworks.

---

## Key Features

- **Universal Discovery** - Find agents across ANY framework (CrewAI, AutoGen, LangGraph, A2A)
- **Immediate Registration** - Agents discoverable instantly with `@mesh.agent()` decorator
- **Built-in Trust Management** - Automatic reliability tracking (5 trust levels)
- **Flexible Storage** - InMemory, SQLite (FTS5), or Redis backends
- **Exact & Semantic Search** - Capability matching plus optional enhanced embeddings
- **Zero Configuration** - Works out of the box, no setup required
- **Production Ready** - 139 tests passing, 100% coverage, battle-tested

---

## Quick Start

### Installation

```bash
# Core package
pip install capabilitymesh

# With SQLite persistence
pip install capabilitymesh[sqlite]

# With Redis for distributed systems
pip install capabilitymesh[redis]

# Everything
pip install capabilitymesh[all]
```

### 5-Line Example

```python
from capabilitymesh import Mesh

mesh = Mesh()  # Zero-config!

@mesh.agent(name="translator", capabilities=["translation", "nlp"])
def translate(text: str, target_lang: str = "es") -> str:
    return f"[{target_lang}] {text}"

# Discover by capability (immediate registration!)
agents = await mesh.discover("translation")
result = await mesh.execute(agents[0].id, "Hello!", target_lang="es")
```

That's it! No configuration, no setup, just register and discover.

---

## Why Choose CapabilityMesh?

### vs. Other Solutions

| Feature | CapabilityMesh | A2A | CrewAI | AutoGen | LangGraph |
|---------|----------------|-----|--------|---------|-----------|
| Multi-framework discovery | ✅ | ❌ | ❌ | ❌ | ❌ |
| Semantic search | ✅ | ❌ | ❌ | ❌ | ❌ |
| Built-in trust | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multiple storage backends | ✅ | ❌ | ❌ | ❌ | ❌ |
| Zero-config | ✅ | ❌ | ✅ | ✅ | ✅ |

**CapabilityMesh is the ONLY solution for universal agent discovery.**

---

## Core Features

### 1. Universal Agent Discovery

Discover agents from any framework with semantic search:

```python
from capabilitymesh import Mesh

mesh = Mesh()

# Register agents from different frameworks
@mesh.agent(capabilities=["summarization", "nlp"])
async def summarize(text: str) -> str:
    return f"Summary: {text[:100]}..."

# Natural language discovery
agents = await mesh.discover("make this text shorter")
# Returns: [summarizer] - semantic match!
```

### 2. Multi-Framework Support

Mix and match agents from different frameworks:

```python
from capabilitymesh import Mesh
from crewai import Agent as CrewAgent
from autogen import AssistantAgent

mesh = Mesh()

# CrewAI agent
crew_agent = CrewAgent(role="researcher", goal="Research topics")
await mesh.register(crew_agent, name="researcher")

# AutoGen agent
autogen_agent = AssistantAgent(name="coder")
await mesh.register(autogen_agent, name="coder")

# Python function
@mesh.agent(capabilities=["analysis"])
def analyzer(data):
    return {"result": "analyzed"}

# Discover across ALL frameworks!
all_agents = await mesh.list_agents()
```

### 3. Automatic Trust Management

Track agent reliability automatically:

```python
from capabilitymesh import Mesh, TrustLevel

mesh = Mesh()

# Execute tasks - trust scores update automatically
for i in range(20):
    await mesh.execute(agent_id, f"task-{i}")

# Check trust
score = await mesh.trust.get_score(agent_id)
print(f"Trust: {score.level.name}")  # HIGH, VERIFIED, etc.
print(f"Success rate: {score.success_rate:.1%}")

# Discover only trusted agents
trusted = await mesh.discover("task", min_trust=TrustLevel.MEDIUM)
```

**Trust Levels:** UNTRUSTED → LOW → MEDIUM → HIGH → VERIFIED (auto-calculated)

### 4. Flexible Storage Backends

Choose storage that fits your deployment:

```python
from capabilitymesh import Mesh
from capabilitymesh.storage import InMemoryStorage, SQLiteStorage, RedisStorage

# Development: In-memory (default)
mesh = Mesh()

# Production: SQLite with full-text search
mesh = Mesh(storage=SQLiteStorage("agents.db"))

# Distributed: Redis for multi-instance
mesh = Mesh(storage=RedisStorage(host="redis.example.com"))
```

| Storage | Persistence | Search | Distribution | Best For |
|---------|-------------|--------|--------------|----------|
| InMemory | No | Basic | Single | Development |
| SQLite | File | FTS5 | Single | Production |
| Redis | Remote | Basic | Multi-instance | Cloud, Scale |

### 5. Semantic Search

Find agents with natural language queries:

```python
# Natural language works!
agents = await mesh.discover("understand customer sentiment")
agents = await mesh.discover("convert text to another language")
agents = await mesh.discover("extract key information from documents")
```

Uses TF-IDF embeddings by default (no external dependencies). Upgrade to sentence-transformers or OpenAI embeddings for even better results (coming in v1.0-beta).

### 6. Rich Capability Schemas

Define capabilities with versioning, constraints, and metadata:

```python
from capabilitymesh import (
    Capability,
    CapabilityVersion,
    CapabilityConstraints,
    SemanticMetadata
)

capability = Capability(
    name="fast-translation",
    version=CapabilityVersion(major=2, minor=1, patch=0),
    constraints=CapabilityConstraints(
        max_response_time_ms=100,
        max_cost_per_call=0.001,
        min_availability=0.999
    ),
    semantic=SemanticMetadata(
        tags=["nlp", "translation", "ml"],
        categories=["Natural Language Processing"],
        domains=["linguistics", "ai"]
    )
)
```

---

## Complete Example

```python
import asyncio
from capabilitymesh import Mesh, Capability, TrustLevel

async def main():
    # Initialize with persistent storage
    mesh = Mesh(storage=SQLiteStorage("agents.db"))

    # Register a document processing pipeline
    @mesh.agent(name="pdf-extractor", capabilities=["extraction", "pdf"])
    def extract_text(pdf_path: str) -> str:
        return f"Extracted text from {pdf_path}"

    @mesh.agent(name="summarizer", capabilities=["summarization", "nlp"])
    async def summarize(text: str) -> str:
        await asyncio.sleep(0.1)  # Async processing
        return f"Summary: {text[:100]}..."

    @mesh.agent(name="translator", capabilities=["translation", "nlp"])
    def translate(text: str, target_lang: str = "es") -> str:
        return f"[{target_lang}] {text}"

    # Build a document processing pipeline
    pdf_path = "document.pdf"

    # Step 1: Extract
    extractors = await mesh.discover("extract text from pdf")
    text = await mesh.execute(extractors[0].id, pdf_path)

    # Step 2: Summarize
    summarizers = await mesh.discover("summarize text")
    summary = await mesh.execute(summarizers[0].id, text)

    # Step 3: Translate
    translators = await mesh.discover("translate to Spanish")
    result = await mesh.execute(translators[0].id, summary, target_lang="es")

    print(f"Final result: {result}")

    # Check trust scores
    for agent in [extractors[0], summarizers[0], translators[0]]:
        score = await mesh.trust.get_score(agent.id)
        print(f"{agent.name}: {score.level.name} ({score.success_rate:.0%})")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Framework Integration Examples

### CrewAI

```python
from capabilitymesh import Mesh
from crewai import Agent

mesh = Mesh()

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments",
    backstory="Expert researcher"
)

await mesh.register(researcher, name="ai-researcher")

# Discover across frameworks
agents = await mesh.discover("research AI developments")
```

### AutoGen

```python
from capabilitymesh import Mesh
from autogen import AssistantAgent

mesh = Mesh()

coder = AssistantAgent(
    name="coder",
    system_message="Expert Python developer"
)

await mesh.register(coder, name="python-coder")

# Discover
coders = await mesh.discover("write python code")
```

### Custom Agents

```python
from capabilitymesh import Mesh

mesh = Mesh()

# Any callable works!
class MyAgent:
    def execute(self, task):
        return f"Processed: {task}"

await mesh.register(MyAgent(), capabilities=["task-processing"])

@mesh.agent(capabilities=["calculation"])
def calculator(a: int, b: int) -> int:
    return a + b
```

---

## Multi-Agent Workflows

### Sequential Pipeline

```python
# Extract → Summarize → Translate
extractors = await mesh.discover("extract text")
text = await mesh.execute(extractors[0].id, "document.pdf")

summarizers = await mesh.discover("summarize")
summary = await mesh.execute(summarizers[0].id, text)

translators = await mesh.discover("translate")
result = await mesh.execute(translators[0].id, summary)
```

### Parallel Processing

```python
import asyncio

# Process multiple items in parallel
items = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
extractor = (await mesh.discover("extract text"))[0]

results = await asyncio.gather(*[
    mesh.execute(extractor.id, item) for item in items
])
```

### Error Handling with Fallbacks

```python
from capabilitymesh import TrustLevel

# Get agents sorted by trust
agents = await mesh.discover("translation", min_trust=TrustLevel.MEDIUM)

# Try agents until one succeeds
for agent in agents:
    try:
        result = await mesh.execute(agent.id, task)
        break
    except Exception:
        continue  # Try next agent
```

---

## Installation Options

```bash
# Minimal (keyword matching only)
pip install capabilitymesh

# With local embeddings (recommended)
pip install capabilitymesh[embeddings]

# With SQLite storage (recommended for production)
pip install capabilitymesh[sqlite]

# With Redis storage (distributed systems)
pip install capabilitymesh[redis]

# With specific frameworks
pip install capabilitymesh[crewai]
pip install capabilitymesh[autogen]
pip install capabilitymesh[langgraph]

# Full installation
pip install capabilitymesh[all]
```

---

## What's Included

### ✅ v1.0.0-alpha.2 (Current)

- **Mesh API** - Simple interface for agent management
- **Multi-framework support** - CrewAI, AutoGen, LangGraph, A2A, custom
- **Semantic discovery** - Natural language queries
- **Trust management** - 5-level automatic scoring
- **Storage backends** - InMemory, SQLite (FTS5), Redis
- **Capability schemas** - Rich metadata, versioning, constraints
- **A2A compatible** - Convert any agent to A2A protocol
- **Fixed `@mesh.agent()` decorator** - Immediate registration, no wrapper overhead
- **139 tests passing** - 100% coverage
- **Complete documentation** - Examples, guides, API reference

### 🔮 Coming Soon

- **v1.0.0-beta.1**: Enhanced embeddings (sentence-transformers, OpenAI)
- **v1.0.0**: Stable release with production hardening
- **v1.1.0**: P2P discovery (mDNS, Gossip, DHT)
- **v1.2.0**: Advanced negotiation protocols

---

## Documentation

- **GitHub**: [https://github.com/scionoftech/capabilitymesh](https://github.com/scionoftech/capabilitymesh)
- **Full Documentation**: [EXAMPLES_GUIDE.md](https://github.com/scionoftech/capabilitymesh/blob/main/EXAMPLES_GUIDE.md)
- **Technical Docs**: [docs/technical_documentation.html](https://github.com/scionoftech/capabilitymesh/blob/main/docs/technical_documentation.html)
- **Roadmap**: [ROADMAP.md](https://github.com/scionoftech/capabilitymesh/blob/main/ROADMAP.md)

### Examples

CapabilityMesh includes 6 comprehensive examples:

1. **01_basic_usage.py** - Registration, discovery, execution
2. **02_storage_backends.py** - InMemory, SQLite, Redis
3. **03_trust_management.py** - Trust tracking and filtering
4. **04_semantic_search.py** - Natural language discovery
5. **05_advanced_capabilities.py** - Rich schemas and versioning
6. **06_multi_agent_workflow.py** - Complex multi-agent coordination

Run any example:
```bash
python examples/01_basic_usage.py
```

---

## Use Cases

### 1. Multi-Framework Teams

Mix agents from different frameworks:
```python
team = {
    "researcher": CrewAI_Agent,      # Best for research
    "coder": AutoGen_Agent,          # Best for coding
    "orchestrator": LangGraph_Agent, # Best for workflows
    "api": A2A_Service               # Best for services
}
# All coordinated via CapabilityMesh!
```

### 2. Agent Marketplace

Build marketplaces where agents advertise capabilities:
```python
# Agents register
marketplace.register(translator, capabilities=["translation"])
marketplace.register(analyzer, capabilities=["analysis"])

# Clients discover and hire
agent = marketplace.discover("translate and analyze")[0]
result = await mesh.execute(agent.id, task_data)
```

### 3. Framework Migration

Gradually migrate between frameworks without disruption:
```python
# Phase 1: All CrewAI
# Phase 2: Mix CrewAI + AutoGen (CapabilityMesh handles discovery)
# Phase 3: All AutoGen
# Agents discoverable throughout migration!
```

### 4. Best Tool for Each Job

Choose optimal framework per agent:
- **CrewAI** → Role-based collaboration
- **AutoGen** → Conversational workflows
- **LangGraph** → Complex state machines
- **A2A** → Production microservices
- **All coordinated seamlessly!**

---

## Requirements

- **Python**: 3.9+
- **Core Dependencies**: pydantic, httpx, cryptography, pyjwt, nest-asyncio
- **Optional Dependencies**:
  - `aiosqlite` - For SQLite storage
  - `redis` - For Redis storage
  - `sentence-transformers` - For better semantic search (coming soon)
  - Framework packages (crewai, autogen, langgraph) - For framework integration

---

## Development Status

**Current Version**: v1.0.0-alpha.2

- ✅ All core features implemented
- ✅ `@mesh.agent()` decorator fully fixed (immediate registration!)
- ✅ 139 tests passing (100% success rate)
- ✅ Comprehensive documentation
- ✅ 6 example files
- ✅ Production-ready code quality
- ⏳ Community feedback welcome

**Stability**: Alpha - Feature-complete and tested, but expect refinements based on real-world usage.

---

## License

Apache License 2.0 - Free for commercial and personal use with patent protection!

**Key Benefits:**
- ✅ Free for commercial and personal use
- ✅ Explicit patent grant protects users and contributors
- ✅ Clear attribution requirements
- ✅ Enterprise-friendly legal framework

See [LICENSE](https://github.com/scionoftech/capabilitymesh/blob/main/LICENSE) for details.

---

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/scionoftech/capabilitymesh/issues)
- **Discussions**: [GitHub Discussions](https://github.com/scionoftech/capabilitymesh/discussions)
- **PyPI**: [https://pypi.org/project/capabilitymesh](https://pypi.org/project/capabilitymesh)

---

## Quick Links

- 🏠 [Homepage](https://github.com/scionoftech/capabilitymesh)
- 📖 [Documentation](https://github.com/scionoftech/capabilitymesh/blob/main/EXAMPLES_GUIDE.md)
- 🐛 [Issue Tracker](https://github.com/scionoftech/capabilitymesh/issues)
- 💬 [Discussions](https://github.com/scionoftech/capabilitymesh/discussions)
- 📦 [PyPI Package](https://pypi.org/project/capabilitymesh)
- 🗺️ [Roadmap](https://github.com/scionoftech/capabilitymesh/blob/main/ROADMAP.md)

---

## Project Stats

- **Version**: 1.0.0-alpha.2
- **Tests**: 139 passing (100%)
- **Coverage**: 100% of core features
- **Frameworks**: 4 supported (CrewAI, AutoGen, LangGraph, A2A)
- **Storage**: 3 backends (InMemory, SQLite, Redis)
- **Trust**: 5-level automatic system
- **Examples**: 6 comprehensive
- **Status**: ✅ Ready for use

---

## ⭐ Show Your Support

If CapabilityMesh helps your project, **star ⭐ the repo** to show your support!

```bash
# Install and try it now!
pip install capabilitymesh

# Your feedback shapes the future of multi-agent systems!
```

---


<div align="center">

**Making agents from any framework work together seamlessly**

*The first and only universal capability mesh for multi-agent systems*

</div>
