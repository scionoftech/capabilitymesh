# CapabilityMesh Examples

This directory contains comprehensive examples demonstrating all features and functionalities of CapabilityMesh, including framework integration, storage backends, trust management, and semantic search.

## New Comprehensive Examples (v1.0)

### [01_basic_usage.py](01_basic_usage.py) - Getting Started ✨
**Status**: ✓ Ready to run (no dependencies)

Master the fundamentals:
- Registering agents with `@mesh.agent()` decorator
- Discovering agents by capability
- Executing agent tasks (sync and async)
- Using Capability objects
- Accessing native functions

```bash
python examples/01_basic_usage.py
```

---

### [02_storage_backends.py](02_storage_backends.py) - Storage Options 💾
**Status**: Partial dependencies (SQLite/Redis optional)

Explore different storage backends:
- **InMemoryStorage** - Fast, non-persistent (default)
- **SQLiteStorage** - Persistent with full-text search
- **RedisStorage** - Distributed, scalable

```bash
# Basic (InMemory only)
python examples/02_storage_backends.py

# With SQLite support
pip install capabilitymesh[sqlite]
python examples/02_storage_backends.py

# With Redis support (requires Redis server)
pip install capabilitymesh[redis]
redis-server  # In another terminal
python examples/02_storage_backends.py
```

**Storage Comparison:**
| Storage    | Persistence  | Search    | Distribution | Best For        |
|------------|--------------|-----------|--------------|-----------------|
| InMemory   | No           | Basic     | Single       | Dev/Test        |
| SQLite     | Yes (file)   | Full-text | Single       | Production      |
| Redis      | Yes (remote) | Basic     | Multi        | Cloud/Scale     |

---

### [03_trust_management.py](03_trust_management.py) - Trust & Reputation 🛡️
**Status**: ✓ Ready to run (no dependencies)

Build reliable multi-agent systems:
- Automatic trust tracking based on execution results
- Manual trust level setting with overrides
- Trust-based agent filtering
- Trust statistics and reporting

```bash
python examples/03_trust_management.py
```

**Trust Levels:**
- `UNTRUSTED (0)` - Never executed or explicitly untrusted
- `LOW (1)` - < 50% success rate OR < 5 executions
- `MEDIUM (2)` - 50-80% success rate, >= 5 executions
- `HIGH (3)` - 80-95% success rate, >= 10 executions
- `VERIFIED (4)` - > 95% success rate, >= 20 executions, OR manually verified

---

### [04_semantic_search.py](04_semantic_search.py) - Semantic Discovery 🔍
**Status**: ✓ Ready to run (no dependencies)

Find agents using natural language:
- Keyword-based embeddings (default, no LLM required)
- Semantic similarity matching
- Category-based discovery
- Adjustable similarity thresholds

```bash
python examples/04_semantic_search.py
```

**How it works:** Agent capabilities are converted to embeddings (vectors), and search queries are matched using cosine similarity.

---

### [05_advanced_capabilities.py](05_advanced_capabilities.py) - Advanced Features 🚀
**Status**: ✓ Ready to run (no dependencies)

Master capability definitions:
- **Structured capabilities** with JSON schemas
- **Capability versioning** and compatibility
- **Constraints** (performance, cost, SLA)
- **Semantic metadata** (tags, categories, domains)

```bash
python examples/05_advanced_capabilities.py
```

**Capability Types:**
- `STRUCTURED` - Well-defined I/O schemas
- `UNSTRUCTURED` - Flexible text-based I/O
- `HYBRID` - Supports both formats

---

### [06_multi_agent_workflow.py](06_multi_agent_workflow.py) - Multi-Agent Coordination 🤝
**Status**: ✓ Ready to run (no dependencies)

Build complex workflows:
- **Sequential pipelines** (A → B → C)
- **Parallel processing** (fan-out/fan-in)
- **Conditional routing** (if/else logic)
- **Error handling** and fallbacks

```bash
python examples/06_multi_agent_workflow.py
```

**Workflow Patterns Demonstrated:**
1. Sequential: Extract → Preprocess → Summarize
2. Parallel: Extract → [Entity, Sentiment, Summary, Translation] → Report
3. Conditional: If language != "en" → Translate → Analyze
4. Fallback: Try Primary → If fails → Use Backup

---

## Original Framework Integration Examples

### 1. `basic_capability.py` - Core Concepts

### 1. `basic_capability.py` - Getting Started
**Status**: ✓ Ready to run (no dependencies)

Learn the fundamentals of CapabilityMesh:
- Creating agent identities with DIDs
- Defining capabilities with semantic versioning
- Version compatibility checking
- Input/output specifications
- Constraint validation

```bash
python examples/basic_capability.py
```

**What you'll learn:**
- How to create an agent identity
- How to define structured and unstructured capabilities
- How version compatibility works
- How to validate capabilities

---

### 2. `framework_integrations.py` - Mock Framework Integration
**Status**: ✓ Ready to run (no dependencies)

Demonstrates the integration API using mock framework agents. This is useful for understanding the integration patterns without installing actual frameworks.

```bash
python examples/framework_integrations.py
```

**Covers:**
- A2A protocol integration and bidirectional conversion
- CrewAI agent wrapping and capability extraction
- AutoGen agent wrapping and dynamic group chat
- LangGraph workflow wrapping and discovery nodes
- Multi-framework coordination

**Note**: Uses mock agents - no framework installation required

---

### 3. `real_world_integrations.py` - Real Framework Integration
**Status**: Requires framework installation

Demonstrates actual framework integration with real agents performing practical tasks.

#### Installation

Install all frameworks at once:
```bash
pip install capabilitymesh[frameworks]
```

Or install individually:
```bash
# For CrewAI examples
pip install crewai langchain-openai

# For AutoGen examples
pip install pyautogen

# For LangGraph examples
pip install langgraph langchain
```

#### API Keys Required

These examples require LLM API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
# OR
export ANTHROPIC_API_KEY="your-anthropic-key"
```

#### Running the Examples

```bash
python examples/real_world_integrations.py
```

**Real-World Scenarios:**

1. **CrewAI Research Team** (`example_crewai_research_team`)
   - 3-agent research team: Researcher, Writer, Editor
   - Realistic roles and backstories
   - Auto-capability extraction from CrewAI metadata
   - Dynamic crew formation

2. **AutoGen Code Review System** (`example_autogen_code_review`)
   - 3 specialized reviewers: Security, Performance, Style
   - Conversational agents with domain expertise
   - Dynamic group chat management
   - Capability-based agent discovery

3. **LangGraph Document Processing** (`example_langgraph_workflow`)
   - Multi-step document analysis pipeline
   - Stateful workflow with 4 processing nodes
   - Keyword extraction, summarization, sentiment analysis, categorization
   - Entire workflow exposed as single capability

4. **Cross-Framework Collaboration** (`example_cross_framework_collaboration`)
   - Content creation pipeline spanning multiple frameworks
   - CrewAI researcher + AutoGen writer + A2A publisher
   - Demonstrates seamless inter-framework discovery
   - Shows universal capability interface in action

---

## Quick Start Guide

### For Beginners

Start with the basic example to understand core concepts:
```bash
python examples/basic_capability.py
```

### Understanding Integration Patterns

Run the mock framework example to see integration patterns:
```bash
python examples/framework_integrations.py
```

### Building Real Applications

1. Install frameworks you want to use:
   ```bash
   pip install capabilitymesh[crewai]  # Just CrewAI
   pip install capabilitymesh[autogen] # Just AutoGen
   pip install capabilitymesh[frameworks] # All frameworks
   ```

2. Set up API keys (for LLM-based frameworks):
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Run real-world examples:
   ```bash
   python examples/real_world_integrations.py
   ```

---

## Example Comparison

| Example | Dependencies | API Keys | Complexity | Best For |
|---------|-------------|----------|------------|----------|
| `basic_capability.py` | None | No | Low | Learning core concepts |
| `framework_integrations.py` | None | No | Medium | Understanding integration API |
| `real_world_integrations.py` | Frameworks | Yes | High | Building real applications |

---

## Code Structure

### basic_capability.py
```python
# 1. Create agent identity
agent_identity = AgentIdentity(...)

# 2. Define capability
capability = Capability(...)

# 3. Check compatibility
is_compatible = capability.is_compatible_with(other_capability)
```

### Framework Integration Pattern
```python
# 1. Import framework-specific adapter
from capabilitymesh.integrations.crewai import ACDPCrewAIAgent

# 2. Create framework agent (CrewAI, AutoGen, etc.)
crew_agent = Agent(role="...", goal="...", backstory="...")

# 3. Wrap with CapabilityMesh
acdp_agent = ACDPCrewAIAgent.wrap(crew_agent)

# 4. Register capabilities (auto-extracted)
capabilities = acdp_agent.register_auto_capabilities()

# 5. Now discoverable by other frameworks!
```

---

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Make sure you're in the project root
cd /path/to/agent_capability_proto

# Set PYTHONPATH
export PYTHONPATH=$PWD  # Linux/Mac
set PYTHONPATH=%CD%     # Windows

# Or install in development mode
pip install -e .
```

### Missing Framework Errors

```
[SKIP] CrewAI not installed: No module named 'crewai'
```

Solution: Install the framework
```bash
pip install crewai  # or pip install capabilitymesh[crewai]
```

### API Key Errors

If you see authentication errors:
```bash
# Set your API key
export OPENAI_API_KEY="sk-your-key-here"

# Verify it's set
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY% # Windows
```

### Unicode/Encoding Errors (Windows)

If you see `UnicodeEncodeError` on Windows:
```bash
# Set encoding before running
set PYTHONIOENCODING=utf-8
python examples/script.py
```

---

## What's Next?

After exploring these examples:

1. **Build Your Own Integration**
   - Create adapters for new frameworks
   - Extend existing integrations
   - Implement custom discovery strategies

2. **Create Multi-Agent Systems**
   - Combine agents from different frameworks
   - Use CapabilityMesh for discovery and negotiation
   - Build complex workflows

3. **Contribute**
   - Share your examples
   - Improve existing integrations
   - Add support for new frameworks

---

## Additional Resources

- **Main Documentation**: `../docs.html`
- **GitHub Repository**: https://github.com/scionoftech/capabilitymesh
- **Integration Architecture**: See `../docs.html#integration-architecture`
- **API Reference**: See `../docs.html#api-reference`

---

## Example Output Samples

### basic_capability.py
```
======================================================================
  Example 1: Creating an Agent Identity
======================================================================

Agent Identity Created:
  DID: did:acdp:abc123...
  Name: TranslatorAgent
  Type: AgentType.LLM
  ...
```

### real_world_integrations.py (with frameworks installed)
```
======================================================================
  Example 1: CrewAI Research Team
======================================================================

Scenario: AI research team analyzing the latest trends...

Wrapping CrewAI agents with CapabilityMesh...

Registered Capabilities:
  Researcher: ai-research-analyst
    - Tags: ['ai', 'research', 'analyst', ...]
  ...

[SUCCESS] CrewAI research team created and registered!
```

---

Built with ❤️ by @scionoftech
