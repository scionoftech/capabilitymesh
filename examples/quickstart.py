"""Quickstart example for CapabilityMesh v1.0.

This example demonstrates the core functionality:
- Registering agents (decorator and explicit)
- Discovering agents by capability
- Executing tasks
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from capabilitymesh import Mesh


async def main():
    print("=" * 60)
    print("CapabilityMesh v1.0 - Quickstart Example")
    print("=" * 60)
    print()

    # Create a mesh (in-memory by default)
    mesh = Mesh()
    print("[OK] Created Mesh with in-memory storage")
    print()

    # ========================================
    # Example 1: Register with decorator
    # ========================================
    print("Example 1: Register agents with decorator")
    print("-" * 60)

    @mesh.agent(name="summarizer", capabilities=["summarization", "text-processing"])
    async def summarize(text: str) -> str:
        """Summarize text into key points."""
        # Simulate summarization
        words = text.split()
        return f"Summary: {len(words)} words - {text[:50]}..."

    @mesh.agent(name="translator", capabilities=["translation", "language"])
    async def translate(text: str, target_lang: str = "es") -> str:
        """Translate text to another language."""
        return f"[{target_lang}] {text}"

    # Agents are registered IMMEDIATELY when decorated - no need to call them first
    print("[OK] Registered 2 agents using @mesh.agent decorator")

    # Verify they're discoverable immediately
    agents = await mesh.discover("summarization")
    print(f"[OK] Found {len(agents)} summarization agent(s) - decorator works!")
    print()

    # ========================================
    # Example 2: Explicit registration
    # ========================================
    print("Example 2: Register agents explicitly")
    print("-" * 60)

    def keyword_extractor(text: str) -> list:
        """Extract keywords from text."""
        return text.lower().split()

    identity = await mesh.register(
        keyword_extractor,
        name="keyword-extractor",
        capabilities=["keyword-extraction", "text-analysis"],
    )

    print(f"[OK] Registered agent: {identity.name}")
    print(f"  Agent ID: {identity.id[:16]}...")
    print()

    # ========================================
    # Example 3: List all agents
    # ========================================
    print("Example 3: List all registered agents")
    print("-" * 60)

    agents = await mesh.list_agents()
    print(f"Total agents registered: {len(agents)}")
    for agent in agents:
        caps = ", ".join([cap.name for cap in agent.capabilities])
        print(f"  • {agent.name}: [{caps}]")
    print()

    # ========================================
    # Example 4: Discover agents by capability
    # ========================================
    print("Example 4: Discover agents by capability")
    print("-" * 60)

    # Discover summarization agents
    results = await mesh.discover("summarization")
    print(f"Agents with 'summarization': {len(results)}")
    for agent_info in results:
        print(f"  • {agent_info.name}")

    # Discover translation agents
    results = await mesh.discover("translation")
    print(f"\nAgents with 'translation': {len(results)}")
    for agent_info in results:
        print(f"  • {agent_info.name}")

    # Discover by partial match
    results = await mesh.discover("text")
    print(f"\nAgents with 'text' (partial match): {len(results)}")
    for agent_info in results:
        print(f"  • {agent_info.name}")
    print()

    # ========================================
    # Example 5: Execute tasks
    # ========================================
    print("Example 5: Execute tasks with discovered agents")
    print("-" * 60)

    # Find summarizer
    summarizers = await mesh.discover("summarization")
    if summarizers:
        agent = summarizers[0]
        result = await mesh.execute(
            agent.id,
            "CapabilityMesh is a universal capability discovery and coordination "
            "layer for multi-agent systems. It enables agents from different "
            "frameworks to discover and work together seamlessly."
        )
        print(f"Summarizer result:")
        print(f"  {result}")

    # Find translator
    translators = await mesh.discover("translation")
    if translators:
        agent = translators[0]
        result = await mesh.execute(agent.id, "Hello, World!", target_lang="es")
        print(f"\nTranslator result:")
        print(f"  {result}")

    # Find keyword extractor
    extractors = await mesh.discover("keyword-extraction")
    if extractors:
        agent = extractors[0]
        result = await mesh.execute(
            agent.id, "Multi-agent systems with capability discovery"
        )
        print(f"\nKeyword extractor result:")
        print(f"  {result}")

    print()

    # ========================================
    # Example 6: Access native agent
    # ========================================
    print("Example 6: Access native agent function")
    print("-" * 60)

    if agents:
        agent = agents[0]
        native_func = await mesh.get_native_async(agent.id)
        print(f"Native function for '{agent.name}': {native_func.__name__}")
        print(f"Function docstring: {native_func.__doc__.strip()}")

    print()
    print("=" * 60)
    print("Quickstart completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
