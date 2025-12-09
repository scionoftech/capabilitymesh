# CapabilityMesh Examples Guide

This guide provides an overview of all examples included in the CapabilityMesh package, demonstrating comprehensive features from basic usage to advanced multi-agent workflows.

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Example Categories](#example-categories)
3. [Running the Examples](#running-the-examples)
4. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
# Basic installation
pip install capabilitymesh

# With SQLite storage support
pip install capabilitymesh[sqlite]

# With Redis storage support
pip install capabilitymesh[redis]

# All optional dependencies
pip install capabilitymesh[all]
```

### Running Your First Example

```bash
python examples/01_basic_usage.py
```

---

## Example Categories

### 🌟 Core Features (Examples 01-03)

These examples cover the fundamental capabilities of CapabilityMesh:

#### 01_basic_usage.py - Getting Started
- **Description**: Learn the basics of agent registration, discovery, and execution
- **Dependencies**: None
- **Key Concepts**:
  - Registering agents with `@mesh.agent()` decorator
  - Discovering agents by capability
  - Executing tasks (sync and async)
  - Working with Capability objects

#### 02_storage_backends.py - Persistence Options
- **Description**: Explore different storage backends for agent persistence
- **Dependencies**: Optional (SQLite, Redis)
- **Storage Types**:
  - InMemoryStorage (default, fast, non-persistent)
  - SQLiteStorage (file-based, full-text search)
  - RedisStorage (distributed, scalable)

#### 03_trust_management.py - Reliability & Trust
- **Description**: Build reliable systems with automatic trust tracking
- **Dependencies**: None
- **Features**:
  - Automatic trust scoring based on execution results
  - Manual trust level overrides
  - Trust-based agent filtering
  - Trust statistics and reporting

---

### 🔍 Advanced Features (Examples 04-06)

These examples demonstrate advanced CapabilityMesh capabilities:

#### 04_semantic_search.py - Natural Language Discovery
- **Description**: Find agents using semantic search and natural language queries
- **Dependencies**: None (uses built-in KeywordEmbedder)
- **Features**:
  - Keyword-based embeddings (no LLM required)
  - Semantic similarity matching
  - Category-based discovery
  - Adjustable similarity thresholds

#### 05_advanced_capabilities.py - Rich Capability Definitions
- **Description**: Master advanced capability features
- **Dependencies**: None
- **Topics Covered**:
  - Structured capabilities with JSON schemas
  - Capability versioning and compatibility
  - Performance and cost constraints
  - Semantic metadata (tags, categories, domains)

#### 06_multi_agent_workflow.py - Agent Coordination
- **Description**: Build complex multi-agent workflows
- **Dependencies**: None
- **Workflow Patterns**:
  - Sequential pipelines (A → B → C)
  - Parallel processing (fan-out/fan-in)
  - Conditional routing (if/else logic)
  - Error handling and fallbacks

---

## Running the Examples

### Prerequisites

Make sure you're in the project root directory:

```bash
cd /path/to/capabilitymesh
```

### Method 1: Direct Execution (Recommended for Development)

The examples include path setup code, so you can run them directly:

```bash
python examples/01_basic_usage.py
python examples/02_storage_backends.py
python examples/03_trust_management.py
python examples/04_semantic_search.py
python examples/05_advanced_capabilities.py
python examples/06_multi_agent_workflow.py
```

### Method 2: After Installation

If you've installed the package:

```bash
pip install -e .  # Install in development mode
python examples/01_basic_usage.py
```

### Running All Examples

Bash/Linux/Mac:
```bash
for file in examples/0*.py; do
    echo "Running $file..."
    python "$file"
    echo "---"
done
```

Windows PowerShell:
```powershell
Get-ChildItem examples\0*.py | ForEach-Object {
    Write-Host "Running $_..."
    python $_.FullName
    Write-Host "---"
}
```

---

## Example-Specific Instructions

### 02_storage_backends.py - Storage Backends

**For SQLite support:**
```bash
pip install capabilitymesh[sqlite]
python examples/02_storage_backends.py
```

**For Redis support:**
```bash
# Install Redis support
pip install capabilitymesh[redis]

# Start Redis server (in separate terminal)
redis-server

# Run example
python examples/02_storage_backends.py
```

### Windows Encoding Issues

If you encounter `UnicodeEncodeError` on Windows:

```bash
# PowerShell
$env:PYTHONIOENCODING="utf-8"
python examples/01_basic_usage.py

# CMD
set PYTHONIOENCODING=utf-8
python examples/01_basic_usage.py
```

---

## Troubleshooting

### Module Not Found Error

**Problem:**
```
ModuleNotFoundError: No module named 'capabilitymesh'
```

**Solutions:**

1. **Install the package:**
   ```bash
   pip install capabilitymesh
   # OR for development
   pip install -e .
   ```

2. **Set PYTHONPATH:**
   ```bash
   # Linux/Mac
   export PYTHONPATH=$PWD
   python examples/01_basic_usage.py

   # Windows CMD
   set PYTHONPATH=%CD%
   python examples\01_basic_usage.py

   # Windows PowerShell
   $env:PYTHONPATH=$PWD
   python examples\01_basic_usage.py
   ```

### Redis Connection Error

**Problem:**
```
Redis connection failed: Error connecting to localhost:6379
```

**Solution:**
Start Redis server:
```bash
# Linux/Mac
redis-server

# Windows (if installed via Chocolatey/MSI)
redis-server.exe

# Or use Docker
docker run -d -p 6379:6379 redis
```

### SQLite Not Available

**Problem:**
```
Note: SQLite storage not available
```

**Solution:**
```bash
pip install capabilitymesh[sqlite]
# or
pip install aiosqlite
```

---

## Key Concepts by Example

### 01_basic_usage.py
- `Mesh()` - Initialize the mesh
- `@mesh.agent()` - Decorator for agent registration
- `mesh.discover()` - Find agents by capability
- `mesh.execute()` - Run agent tasks

### 02_storage_backends.py
- `InMemoryStorage()` - Default, fast, non-persistent
- `SQLiteStorage(db_path)` - File-based persistence
- `RedisStorage(host, port)` - Distributed storage

### 03_trust_management.py
- `TrustLevel` - Enum (UNTRUSTED → LOW → MEDIUM → HIGH → VERIFIED)
- `mesh.trust.record_execution()` - Track results
- `mesh.trust.set_level()` - Manual override
- `mesh.discover(min_trust=...)` - Filter by trust

### 04_semantic_search.py
- `KeywordEmbedder` - Default embedder (no LLM)
- `cosine_similarity()` - Calculate similarity
- `mesh.discover(query, min_similarity=...)` - Semantic search

### 05_advanced_capabilities.py
- `CapabilityType` - STRUCTURED, UNSTRUCTURED, HYBRID
- `CapabilityVersion` - Semantic versioning
- `CapabilityConstraints` - Performance, cost, SLA
- `SemanticMetadata` - Tags, categories, domains

### 06_multi_agent_workflow.py
- Sequential: `result1 = await execute(); result2 = await execute(result1)`
- Parallel: `await asyncio.gather(execute1(), execute2(), execute3())`
- Conditional: `if condition: await execute(agentA) else: await execute(agentB)`
- Fallback: `try: await execute(primary) except: await execute(fallback)`

---

## Next Steps

After exploring these examples:

1. **Build Your Own**
   - Start with `01_basic_usage.py` as a template
   - Add your own agents and capabilities
   - Experiment with different workflows

2. **Explore Framework Integration**
   - See `examples/framework_integrations.py` for CrewAI, AutoGen, LangGraph
   - Check `examples/real_world_integrations.py` for production examples

3. **Read the Documentation**
   - API Reference: [docs/api_reference.md](docs/api_reference.md)
   - Architecture Guide: [docs/architecture.md](docs/architecture.md)
   - Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

4. **Join the Community**
   - GitHub Issues: https://github.com/scionoftech/capabilitymesh/issues
   - Discussions: https://github.com/scionoftech/capabilitymesh/discussions
   - Contributing Guide: See CONTRIBUTING.md

---

## Feature Matrix

| Example | Registration | Discovery | Execution | Storage | Trust | Semantic | Workflows |
|---------|-------------|-----------|-----------|---------|-------|----------|-----------|
| 01 | ✅ | ✅ | ✅ | - | - | - | - |
| 02 | ✅ | ✅ | - | ✅ | - | - | - |
| 03 | ✅ | ✅ | ✅ | - | ✅ | - | - |
| 04 | ✅ | ✅ | - | - | - | ✅ | - |
| 05 | ✅ | - | - | - | - | ✅ | - |
| 06 | ✅ | ✅ | ✅ | - | - | - | ✅ |

---

## Additional Resources

- **Week 3 Implementation**: Storage backends and trust management were added in v1.0-alpha.1
- **Test Suite**: See `tests/unit/test_trust.py` and `tests/unit/test_sqlite_storage.py` for more usage examples
- **API Examples**: Check integration tests in `tests/integration/` for framework-specific examples

---

Built with ❤️ by the CapabilityMesh Team
