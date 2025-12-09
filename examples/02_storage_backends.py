"""Example 2: Storage Backends

This example demonstrates different storage options:
- InMemoryStorage (default, fast, non-persistent)
- SQLiteStorage (persistent, full-text search)
- RedisStorage (distributed, scalable)
"""

import asyncio
from capabilitymesh import Mesh, Capability, AgentType
from capabilitymesh.storage import InMemoryStorage

# Optional imports (require extra dependencies)
try:
    from capabilitymesh.storage import SQLiteStorage
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    print("Note: SQLite storage not available. Install with: pip install capabilitymesh[sqlite]")

try:
    from capabilitymesh.storage import RedisStorage
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Note: Redis storage not available. Install with: pip install capabilitymesh[redis]")


async def demo_in_memory_storage():
    """Demonstrate in-memory storage (default)."""
    print("\n" + "=" * 60)
    print("1. InMemoryStorage (Default)")
    print("=" * 60)

    # Create mesh with default in-memory storage
    mesh = Mesh()

    # Register an agent
    @mesh.agent(name="calculator", capabilities=["math", "calculation"])
    def calculate(expression: str) -> float:
        """Evaluate mathematical expression."""
        return eval(expression)  # Note: eval is unsafe in production

    print("✓ Registered calculator agent")

    # Discover and execute
    agents = await mesh.discover("math")
    print(f"✓ Found {len(agents)} math agents")

    result = await mesh.execute(agents[0].id, "2 + 2 * 3")
    print(f"✓ Calculation result: 2 + 2 * 3 = {result}")

    print("\nFeatures:")
    print("  - Fast (in-memory)")
    print("  - Simple (no setup required)")
    print("  - Non-persistent (data lost on restart)")
    print("  - Best for: Development, testing, single-process apps")


async def demo_sqlite_storage():
    """Demonstrate SQLite storage."""
    if not SQLITE_AVAILABLE:
        print("\n⏭  Skipping SQLite demo (not installed)")
        return

    print("\n" + "=" * 60)
    print("2. SQLiteStorage (Persistent)")
    print("=" * 60)

    # Create mesh with SQLite storage
    storage = SQLiteStorage(db_path="capabilitymesh_example.db")
    mesh = Mesh(storage=storage)

    # Register agents
    @mesh.agent(name="text-processor", capabilities=["text", "processing"])
    def process_text(text: str) -> str:
        """Process text (uppercase)."""
        return text.upper()

    @mesh.agent(name="word-counter", capabilities=["text", "analysis"])
    def count_words(text: str) -> int:
        """Count words in text."""
        return len(text.split())

    print("✓ Registered 2 text agents")

    # Full-text search
    agents = await mesh.discover("text processing")
    print(f"✓ Full-text search found {len(agents)} agents")
    for agent in agents:
        print(f"  - {agent.name}")

    # Execute
    if agents:
        result = await mesh.execute(agents[0].id, "hello world")
        print(f"✓ Processing result: {result}")

    # Close storage
    await storage.close()

    print("\nFeatures:")
    print("  - Persistent (survives restarts)")
    print("  - Full-text search (FTS5)")
    print("  - File-based (single file)")
    print("  - Best for: Production, single-server apps, edge devices")

    # Demonstrate persistence
    print("\n  Demonstrating persistence...")
    storage2 = SQLiteStorage(db_path="capabilitymesh_example.db")
    mesh2 = Mesh(storage=storage2)
    agents2 = await mesh2.list()
    print(f"  ✓ Loaded {len(agents2)} agents from disk")
    await storage2.close()


async def demo_redis_storage():
    """Demonstrate Redis storage."""
    if not REDIS_AVAILABLE:
        print("\n⏭  Skipping Redis demo (not installed)")
        return

    print("\n" + "=" * 60)
    print("3. RedisStorage (Distributed)")
    print("=" * 60)

    try:
        # Create mesh with Redis storage
        storage = RedisStorage(
            host="localhost",
            port=6379,
            db=0,
            prefix="example:",
            ttl=3600,  # 1 hour TTL
        )

        mesh = Mesh(storage=storage)

        # Register agents
        @mesh.agent(name="api-caller", capabilities=["api", "http"])
        async def call_api(url: str) -> dict:
            """Call external API."""
            await asyncio.sleep(0.1)  # Simulate API call
            return {"status": "success", "url": url}

        @mesh.agent(name="cache-manager", capabilities=["cache", "storage"])
        def manage_cache(key: str, value: str) -> str:
            """Manage cache entries."""
            return f"Cached: {key} = {value}"

        print("✓ Registered 2 agents to Redis")

        # Discover
        agents = await mesh.discover("api")
        print(f"✓ Found {len(agents)} API agents")

        # Execute
        if agents:
            result = await mesh.execute(agents[0].id, "https://api.example.com")
            print(f"✓ API call result: {result}")

        # List all agents
        all_agents = await mesh.list()
        print(f"✓ Total agents in Redis: {len(all_agents)}")

        # Close storage
        await storage.close()

        print("\nFeatures:")
        print("  - Distributed (multi-server)")
        print("  - Scalable (handles large registries)")
        print("  - TTL support (automatic expiration)")
        print("  - Best for: Microservices, cloud deployments, high availability")

    except Exception as e:
        print(f"⚠  Redis connection failed: {e}")
        print("   Make sure Redis server is running: redis-server")


async def demo_storage_comparison():
    """Compare storage backends."""
    print("\n" + "=" * 60)
    print("Storage Backend Comparison")
    print("=" * 60)

    print("\n┌─────────────┬──────────────┬───────────┬──────────────┬─────────────┐")
    print("│ Storage     │ Persistence  │ Search    │ Distribution │ Best For    │")
    print("├─────────────┼──────────────┼───────────┼──────────────┼─────────────┤")
    print("│ InMemory    │ No           │ Basic     │ Single       │ Dev/Test    │")
    print("│ SQLite      │ Yes (file)   │ Full-text │ Single       │ Production  │")
    print("│ Redis       │ Yes (remote) │ Basic     │ Multi        │ Cloud/Scale │")
    print("└─────────────┴──────────────┴───────────┴──────────────┴─────────────┘")

    print("\nInstallation:")
    print("  InMemory:  pip install capabilitymesh")
    print("  SQLite:    pip install capabilitymesh[sqlite]")
    print("  Redis:     pip install capabilitymesh[redis]")


async def main():
    print("=" * 60)
    print("CapabilityMesh - Storage Backends Example")
    print("=" * 60)

    await demo_in_memory_storage()
    await demo_sqlite_storage()
    await demo_redis_storage()
    await demo_storage_comparison()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
