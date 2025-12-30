 # Discovery Architecture in CapabilityMesh

This document explains how agent discovery works in CapabilityMesh, its current capabilities, limitations, and future roadmap.

## Table of Contents

- [Overview](#overview)
- [Current Architecture (v1.0.0-alpha.1)](#current-architecture-v100-alpha1)
- [Discovery vs. Execution](#discovery-vs-execution)
- [Cross-Process Discovery](#cross-process-discovery)
- [Distributed Execution Patterns](#distributed-execution-patterns)
- [Future: True P2P Discovery](#future-true-p2p-discovery-v110)
- [Use Case Matrix](#use-case-matrix)
- [Best Practices](#best-practices)

---

## Overview

**CapabilityMesh Discovery** enables agents to find other agents based on their capabilities. The discovery mechanism answers the question: *"Which agents can perform task X?"*

However, understanding the **scope of discovery** is crucial:

- **Discovery** = Finding agents and their metadata
- **Execution** = Actually running/calling those agents

These are **separate concerns** with different architectural constraints.

---

## Current Architecture (v1.0.0-alpha.1)

### Component Scope

| Component | Scope | Persisted? | Shareable Across Processes? |
|-----------|-------|------------|----------------------------|
| **Storage (Redis/SQLite)** | Multi-process | Yes | ✅ Yes |
| **Function Registry** | Single-process | No (in-memory) | ❌ No |
| **Embeddings Cache** | Single-process | No (in-memory) | ❌ No |
| **Trust Scores** | Multi-process (w/ Redis) | Yes | ✅ Yes |
| **Agent Metadata** | Multi-process (w/ Redis) | Yes | ✅ Yes |

### What Gets Stored in Shared Storage (Redis/SQLite)?

When you register an agent with shared storage:

```python
mesh = Mesh(storage=RedisStorage(host="localhost", port=6379))

@mesh.agent(name="translator", capabilities=["translation"])
def translate(text: str, target_lang: str = "es") -> str:
    return f"[{target_lang}]: {text}"
```

**Stored in Redis:**
- ✅ Agent ID
- ✅ Agent name (`"translator"`)
- ✅ Capabilities (`["translation"]`)
- ✅ Agent type (SOFTWARE, LLM, HUMAN)
- ✅ Metadata
- ✅ Registration timestamp
- ✅ Trust scores (success/failure counts)

**NOT stored in Redis:**
- ❌ The actual Python function (`translate`)
- ❌ Capability embeddings (vectors)
- ❌ Execution context

### What Gets Stored Locally (In-Memory)?

Each Mesh instance maintains:

```python
class Mesh:
    def __init__(self, ...):
        self._function_registry: Dict[str, Callable] = {}        # Python functions
        self._capability_embeddings: Dict[str, List[float]] = {} # Embedding vectors
```

**Local to process:**
- The actual callable functions (Python code)
- Embedding vectors for semantic search
- Framework-specific agent objects (CrewAI, AutoGen instances)

---

## Discovery vs. Execution

### Discovery: What You CAN Do Across Processes

**Scenario**: Two separate processes with shared Redis storage

```python
# ============================================
# PROCESS A (service-a.py) - Translation Service
# ============================================
from capabilitymesh import Mesh
from capabilitymesh.storage import RedisStorage

mesh_a = Mesh(storage=RedisStorage(host="redis-server", port=6379))

@mesh_a.agent(name="translator", capabilities=["translation", "nlp"])
def translate(text: str, target_lang: str = "es") -> str:
    return f"[{target_lang}]: {text}"

@mesh_a.agent(name="summarizer", capabilities=["summarization", "nlp"])
def summarize(text: str) -> str:
    return f"Summary: {text[:100]}"

print("Service A: Agents registered")
```

```python
# ============================================
# PROCESS B (service-b.py) - Orchestrator Service
# ============================================
from capabilitymesh import Mesh
from capabilitymesh.storage import RedisStorage

mesh_b = Mesh(storage=RedisStorage(host="redis-server", port=6379))

# ✅ DISCOVERY WORKS - Can see agents from Process A
agents = await mesh_b.discover("translation")
print(f"Found {len(agents)} agents")  # Output: Found 1 agents
print(f"Agent name: {agents[0].name}")  # Output: Agent name: translator
print(f"Capabilities: {agents[0].capabilities}")  # Output: Capabilities: [translation, nlp]

# List all agents (including from other processes)
all_agents = await mesh_b.list_agents()
print(f"Total agents: {len(all_agents)}")  # Output: Total agents: 2 (translator + summarizer)
```

**✅ This works!** Process B can discover agents registered by Process A through shared Redis storage.

### Execution: What You CANNOT Do Across Processes

```python
# ============================================
# PROCESS B (continued) - Trying to Execute
# ============================================

# ❌ EXECUTION FAILS - Function is not in Process B's memory
try:
    result = await mesh_b.execute(agents[0].id, "Hello world!")
    print(result)
except ExecutionError as e:
    print(f"Error: {e}")
    # Output: Error: Agent not found in function registry
```

**❌ This fails!** Why?

1. Process B discovered the agent metadata from Redis
2. But the actual Python function `translate()` only exists in Process A's memory
3. Process B's `_function_registry` doesn't have the function
4. `mesh_b.execute()` looks for the function locally and fails

---

## Cross-Process Discovery

### Pattern 1: Shared Discovery with Local Execution

**Use Case**: Multiple services need to know what agents exist, but each executes its own agents.

```python
# ============================================
# SERVICE A - Translation Service
# ============================================
mesh_a = Mesh(storage=RedisStorage())

@mesh_a.agent(name="translator", capabilities=["translation"])
def translate(text: str) -> str:
    return f"Translated: {text}"

# Register and keep running
while True:
    # Wait for work...
    # Can execute its own agents
    pass

# ============================================
# SERVICE B - Summarization Service
# ============================================
mesh_b = Mesh(storage=RedisStorage())

@mesh_b.agent(name="summarizer", capabilities=["summarization"])
def summarize(text: str) -> str:
    return f"Summary: {text[:100]}"

# Both services can discover each other
all_agents = await mesh_b.list_agents()
# Returns: [translator (from A), summarizer (from B)]

# But each executes only its own agents
# Service B executes its own summarizer
summarizer_agents = await mesh_b.discover("summarization")
result = await mesh_b.execute(summarizer_agents[0].id, "Long text...")  # ✅ Works
```

**Benefits**:
- Shared agent registry (visibility)
- Each service knows what capabilities exist in the system
- Trust scores are shared (everyone sees reliability metrics)

**Limitations**:
- Can't execute remote agents directly
- Need additional coordination (message queues, RPC) for cross-service calls

### Pattern 2: Discovery + Message Queue Coordination

```python
# ============================================
# SERVICE A - Translation Service
# ============================================
import redis.asyncio as redis
from capabilitymesh import Mesh
from capabilitymesh.storage import RedisStorage

mesh = Mesh(storage=RedisStorage())
pubsub = redis.Redis().pubsub()

@mesh.agent(name="translator", capabilities=["translation"])
def translate(text: str, target_lang: str = "es") -> str:
    return f"[{target_lang}]: {text}"

# Listen for translation requests
await pubsub.subscribe('translation_requests')
async for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])

        # Execute locally
        result = await mesh.execute(data['agent_id'], data['text'], **data['params'])

        # Publish result
        await redis.publish('translation_results', json.dumps({
            'request_id': data['request_id'],
            'result': result
        }))

# ============================================
# SERVICE B - Orchestrator
# ============================================
mesh_b = Mesh(storage=RedisStorage())

# Discover translator
agents = await mesh_b.discover("translation")
translator_id = agents[0].id

# Send execution request via message queue
request_id = str(uuid4())
await redis.publish('translation_requests', json.dumps({
    'request_id': request_id,
    'agent_id': translator_id,
    'text': 'Hello world!',
    'params': {'target_lang': 'fr'}
}))

# Wait for result
await pubsub.subscribe('translation_results')
# ... handle result ...
```

**Benefits**:
- Decoupled services
- Async execution
- Load balancing possible

**Drawbacks**:
- More complex architecture
- Manual coordination needed

---

## Distributed Execution Patterns

### A2A Agents: The Exception

**A2A (Agent-to-Agent) protocol agents are truly distributed** because they store a **URL** instead of a Python function.

```python
# ============================================
# SERVICE A - Runs HTTP server at http://service-a:8000
# ============================================
from fastapi import FastAPI

app = FastAPI()

@app.post("/api/translate")
async def translate_endpoint(request: dict):
    text = request.get("text", "")
    target_lang = request.get("target_lang", "es")
    return {"result": f"[{target_lang}]: {text}"}

# Run: uvicorn service_a:app --host 0.0.0.0 --port 8000

# ============================================
# SERVICE B - Registers A2A agent and executes remotely
# ============================================
from capabilitymesh import Mesh
from capabilitymesh.integrations.a2a import A2AAdapter
from capabilitymesh.storage import RedisStorage

mesh = Mesh(storage=RedisStorage())

# Wrap A2A HTTP endpoint
translator = A2AAdapter.wrap(
    agent_url="http://service-a:8000/api/translate",
    name="http-translator"
)

# Register with capabilities
await mesh.register(translator, capabilities=["translation"])

# ============================================
# SERVICE C - Discovers and executes remotely
# ============================================
mesh_c = Mesh(storage=RedisStorage())

# Discover (finds the A2A agent)
agents = await mesh_c.discover("translation")

# ✅ EXECUTION WORKS - Makes HTTP call to service-a:8000
result = await mesh_c.execute(
    agents[0].id,
    {"text": "Hello world!", "target_lang": "fr"}
)
print(result)  # {"result": "[fr]: Hello world!"}
```

**Why this works**:
- Agent metadata in Redis includes the URL: `http://service-a:8000/api/translate`
- `execute()` makes an HTTP POST request instead of calling a local function
- Any service with network access can execute the agent

### Microservices Pattern

Combine Redis discovery with A2A execution for distributed systems:

```python
# Service Architecture:
#
# ┌─────────────────┐      ┌─────────────────┐
# │  Translation    │      │  Summarization  │
# │  Service        │      │  Service        │
# │  (HTTP/A2A)     │      │  (HTTP/A2A)     │
# └────────┬────────┘      └────────┬────────┘
#          │                        │
#          └────────┬───────────────┘
#                   │
#            ┌──────▼──────┐
#            │    Redis    │
#            │  (Discovery)│
#            └──────┬──────┘
#                   │
#          ┌────────▼────────┐
#          │  Orchestrator   │
#          │  Service        │
#          └─────────────────┘

# Each microservice:
# 1. Registers its A2A endpoint with CapabilityMesh + Redis
# 2. Can discover other services
# 3. Can execute remote services via HTTP

# Orchestrator can:
# - Discover all available services
# - Chain them together in workflows
# - Track trust scores across the system
```

---

## Future: True P2P Discovery (v1.1.0+)

The roadmap includes **peer-to-peer discovery** that works without shared storage:

### Planned Architecture

```python
from capabilitymesh import Mesh
from capabilitymesh.discovery import DiscoveryEngine, DiscoveryTier

# Initialize with P2P discovery
mesh = Mesh(
    discovery=DiscoveryEngine(
        tiers=[
            DiscoveryTier.LOCAL,    # mDNS - local network discovery
            DiscoveryTier.CLUSTER,  # Gossip protocol - cluster discovery
            DiscoveryTier.GLOBAL,   # DHT (Kademlia) - global discovery
        ]
    )
)

# Agent automatically advertises on the network
@mesh.agent(name="translator", capabilities=["translation"])
def translate(text: str) -> str:
    return f"Translated: {text}"

# Discovery works across machines without Redis!
# Agents broadcast via:
# - mDNS on local network (like Bonjour/Avahi)
# - Gossip protocol for cluster coordination
# - Distributed Hash Table for internet-wide discovery
```

### P2P Discovery Tiers

| Tier | Protocol | Scope | Use Case |
|------|----------|-------|----------|
| **LOCAL** | mDNS (Multicast DNS) | Same subnet | Development, edge devices |
| **CLUSTER** | Gossip Protocol | Data center | Kubernetes cluster, corporate network |
| **GLOBAL** | DHT (Kademlia) | Internet | Public agent marketplace |

### How It Will Work

```python
# Machine A (192.168.1.10)
mesh_a = Mesh(discovery=DiscoveryEngine(tiers=[DiscoveryTier.LOCAL]))

@mesh_a.agent(name="translator", capabilities=["translation"])
def translate(text: str) -> str:
    return f"Translated: {text}"

# Agent broadcasts via mDNS: "translator.capabilitymesh.local"

# Machine B (192.168.1.20) - Same network
mesh_b = Mesh(discovery=DiscoveryEngine(tiers=[DiscoveryTier.LOCAL]))

# Discovers via mDNS multicast
agents = await mesh_b.discover("translation")
# Finds: translator @ 192.168.1.10:5000

# Execution via RPC/gRPC/HTTP
result = await mesh_b.execute(agents[0].id, "Hello")  # Network call to Machine A
```

**Benefits**:
- No central registry needed
- Auto-discovery on local networks
- Fault-tolerant (gossip handles node failures)
- Scalable (DHT for large networks)

---

## Use Case Matrix

### What Works Today (v1.0.0-alpha.1)

| Scenario | Discovery | Execution | Storage | Notes |
|----------|-----------|-----------|---------|-------|
| **Single Process, Multiple Frameworks** | ✅ | ✅ | In-Memory | Best for monolithic apps |
| **Single Process, Persistent Storage** | ✅ | ✅ | SQLite | Survives restarts |
| **Multi-Process Discovery Only** | ✅ | ❌ | Redis | Can see agents, can't execute |
| **Multi-Process via A2A** | ✅ | ✅ | Redis | HTTP-based execution |
| **Microservices with A2A** | ✅ | ✅ | Redis | Production pattern |
| **Multi-Process + Message Queue** | ✅ | ✅ (manual) | Redis | Custom coordination |

### Legend
- ✅ = Works out of the box
- ❌ = Not supported
- ✅ (manual) = Requires custom implementation

### Example: Single Process, Multiple Frameworks

```python
# ✅ Perfect use case for v1.0
from capabilitymesh import Mesh
from crewai import Agent as CrewAgent
from autogen import AssistantAgent

mesh = Mesh()  # In-memory storage

# Mix frameworks in one app
crew_agent = CrewAgent(role="researcher", goal="Research topics")
await mesh.register(crew_agent, name="researcher")

autogen_agent = AssistantAgent(name="coder", system_message="Write code")
await mesh.register(autogen_agent, name="coder")

@mesh.agent(capabilities=["translation"])
def translator(text: str) -> str:
    return f"Translated: {text}"

# Discover across frameworks
agents = await mesh.discover("research")  # Finds crew_agent
coders = await mesh.discover("code")       # Finds autogen_agent
translators = await mesh.discover("translation")  # Finds translator

# Execute any agent
result = await mesh.execute(agents[0].id, "AI trends")  # ✅ Works
```

### Example: Microservices with A2A

```python
# ✅ Production-ready distributed pattern

# Service 1: Translation (Port 8001)
app = FastAPI()

@app.post("/translate")
async def translate(request: dict):
    return {"result": f"Translated: {request['text']}"}

mesh = Mesh(storage=RedisStorage())
translator = A2AAdapter.wrap("http://localhost:8001/translate")
await mesh.register(translator, capabilities=["translation"])

# Service 2: Summarization (Port 8002)
app = FastAPI()

@app.post("/summarize")
async def summarize(request: dict):
    return {"result": f"Summary: {request['text'][:100]}"}

mesh = Mesh(storage=RedisStorage())
summarizer = A2AAdapter.wrap("http://localhost:8002/summarize")
await mesh.register(summarizer, capabilities=["summarization"])

# Service 3: Orchestrator
mesh = Mesh(storage=RedisStorage())

# Discover both services
translators = await mesh.discover("translation")
summarizers = await mesh.discover("summarization")

# Execute remotely via HTTP
translation = await mesh.execute(translators[0].id, {"text": "Hello"})
summary = await mesh.execute(summarizers[0].id, {"text": "Long text..."})

# ✅ All works - HTTP-based execution
```

---

## Best Practices

### For Single-Process Applications

**Use InMemoryStorage or SQLiteStorage**:
```python
# Development
mesh = Mesh()  # In-memory, fast

# Production (single instance)
mesh = Mesh(storage=SQLiteStorage("agents.db"))  # Persists across restarts
```

**Mix frameworks freely**:
```python
# All in one process - works perfectly
mesh.register(crewai_agent)
mesh.register(autogen_agent)
mesh.register(python_function)
mesh.register(async_function)
```

### For Distributed Systems

**Option 1: Use A2A Pattern (Recommended)**:
```python
# Each service exposes HTTP endpoints
# Register with A2AAdapter
# Execution happens via HTTP
```

**Option 2: Shared Discovery with Manual Execution**:
```python
# Use Redis for discovery
# Implement your own RPC/message queue for execution
# Best when you already have messaging infrastructure
```

**Option 3: Hybrid (Future)**:
```python
# Wait for v1.1.0+ with P2P discovery
# Will support gRPC/MQTT transports
# Native distributed execution
```

### For Development

**Start simple, scale later**:
```python
# Phase 1: Single process, in-memory
mesh = Mesh()

# Phase 2: Persist agents
mesh = Mesh(storage=SQLiteStorage("agents.db"))

# Phase 3: Multiple processes, shared discovery
mesh = Mesh(storage=RedisStorage())

# Phase 4: Microservices with A2A
# Convert functions to HTTP services
# Use A2AAdapter for distributed execution
```

### Trust and Discovery Filters

Even across processes, trust scores work with Redis:

```python
# Service A registers and executes agents
mesh_a = Mesh(storage=RedisStorage())

@mesh_a.agent(capabilities=["task"])
def unreliable_agent(x):
    if random.random() < 0.5:
        raise ValueError("Failed!")
    return "success"

# Execute multiple times - trust score updates in Redis
for i in range(20):
    try:
        await mesh_a.execute(agent_id, f"task-{i}")
    except:
        pass

# Service B discovers with trust filter
mesh_b = Mesh(storage=RedisStorage())

# Only get reliable agents (trust scores shared via Redis)
reliable_agents = await mesh_b.discover("task", min_trust=TrustLevel.MEDIUM)
```

---

## Summary

### Current Capabilities (v1.0.0-alpha.1)

**✅ Fully Supported**:
- Single-process multi-framework coordination
- Distributed discovery via Redis/SQLite
- A2A-based distributed execution
- Cross-process trust tracking (with Redis)
- Semantic search and capability matching

**⚠️ Limitations**:
- Python functions can't execute across processes
- Embeddings are not shared (each process computes its own)
- Framework agents (CrewAI/AutoGen) are process-local

**🔮 Future (v1.1.0+)**:
- True P2P discovery (mDNS, Gossip, DHT)
- gRPC/MQTT transports
- Distributed execution for Python functions
- Shared embedding cache

### Decision Tree

```
Do you need distributed execution?
│
├─ No → Use Mesh() with InMemoryStorage or SQLiteStorage
│        Perfect for single-process multi-framework apps
│
└─ Yes → Are your agents already HTTP services?
         │
         ├─ Yes → Use A2AAdapter with RedisStorage
         │         Distributed discovery + execution works today
         │
         └─ No → Options:
                 1. Convert to HTTP services + A2A (recommended)
                 2. Use Redis for discovery + custom RPC
                 3. Wait for v1.1.0+ P2P discovery
```

---

## Resources

- **Examples**: See `examples/02_storage_backends.py` for Redis/SQLite usage
- **A2A Integration**: See `examples/framework_integrations.py` for A2A adapter
- **Roadmap**: See `ROADMAP.md` for P2P discovery timeline
- **Storage Backends**: See `capabilitymesh/storage/` for implementation details
