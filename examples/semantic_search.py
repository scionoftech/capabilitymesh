"""Semantic search example for CapabilityMesh v1.0.

This example demonstrates semantic search capabilities:
- Natural language queries
- Similarity-based ranking
- Better than keyword matching
- Different embedder options
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from capabilitymesh import Mesh, KeywordEmbedder


async def main():
    print("=" * 70)
    print("CapabilityMesh v1.0 - Semantic Search Example")
    print("=" * 70)
    print()

    # Create mesh with default embedder (KeywordEmbedder - no dependencies)
    mesh = Mesh(embedder=KeywordEmbedder(dimension=512))
    print("[OK] Created Mesh with KeywordEmbedder")
    print()

    # ========================================
    # Register diverse agents
    # ========================================
    print("Registering agents with various capabilities...")
    print("-" * 70)

    agents = [
        {
            "name": "data-analyzer",
            "capabilities": ["data-analysis", "statistics", "visualization"],
            "func": lambda x: f"Analyzing: {x}",
        },
        {
            "name": "text-summarizer",
            "capabilities": ["text-summarization", "content-extraction", "nlp"],
            "func": lambda x: f"Summarizing: {x}",
        },
        {
            "name": "language-translator",
            "capabilities": ["translation", "localization", "multilingual"],
            "func": lambda x: f"Translating: {x}",
        },
        {
            "name": "image-processor",
            "capabilities": ["image-processing", "computer-vision", "ocr"],
            "func": lambda x: f"Processing image: {x}",
        },
        {
            "name": "code-reviewer",
            "capabilities": ["code-review", "static-analysis", "security-scan"],
            "func": lambda x: f"Reviewing code: {x}",
        },
        {
            "name": "sentiment-analyzer",
            "capabilities": ["sentiment-analysis", "emotion-detection", "nlp"],
            "func": lambda x: f"Analyzing sentiment: {x}",
        },
        {
            "name": "recommendation-engine",
            "capabilities": ["recommendations", "personalization", "ml"],
            "func": lambda x: f"Recommending: {x}",
        },
        {
            "name": "anomaly-detector",
            "capabilities": ["anomaly-detection", "monitoring", "alerting"],
            "func": lambda x: f"Detecting anomalies: {x}",
        },
    ]

    for agent_info in agents:
        await mesh.register(
            agent_info["func"],
            name=agent_info["name"],
            capabilities=agent_info["capabilities"],
        )
        print(f"  [OK] {agent_info['name']}")

    print(f"\nTotal registered: {len(agents)} agents")
    print()

    # ========================================
    # Example 1: Natural language queries
    # ========================================
    print("Example 1: Natural Language Queries")
    print("-" * 70)

    queries = [
        "I need to shorten this long article",
        "convert text to Spanish",
        "find patterns in my dataset",
        "check my Python code for bugs",
        "understand the mood of customer feedback",
        "suggest products users might like",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = await mesh.discover(query, limit=3, min_similarity=0.05)

        if results:
            print(f"  Found {len(results)} matches:")
            for i, agent_info in enumerate(results, 1):
                caps = ", ".join([cap.name for cap in agent_info.capabilities[:2]])
                print(f"    {i}. {agent_info.name} ({caps})")
        else:
            print("  No matches found")

    print()

    # ========================================
    # Example 2: Similarity thresholds
    # ========================================
    print("Example 2: Adjusting Similarity Thresholds")
    print("-" * 70)

    query = "analyze customer reviews"
    print(f"Query: '{query}'")
    print()

    thresholds = [0.0, 0.05, 0.1, 0.2]
    for threshold in thresholds:
        results = await mesh.discover(query, limit=10, min_similarity=threshold)
        print(f"  min_similarity={threshold:0.2f}: {len(results)} results")
        if results and threshold >= 0.05:
            for agent_info in results[:3]:
                print(f"    - {agent_info.name}")

    print()

    # ========================================
    # Example 3: Semantic vs Keyword
    # ========================================
    print("Example 3: Semantic Understanding")
    print("-" * 70)
    print("Semantic search understands related concepts, not just keywords:")
    print()

    semantic_queries = [
        ("translate Japanese to English", ["language-translator"]),
        ("extract key points from document", ["text-summarizer", "data-analyzer"]),
        ("detect unusual patterns", ["anomaly-detector", "data-analyzer"]),
        ("scan code for vulnerabilities", ["code-reviewer"]),
        ("understand customer emotions", ["sentiment-analyzer"]),
    ]

    for query, expected_types in semantic_queries:
        results = await mesh.discover(query, limit=2, min_similarity=0.05)
        print(f"Query: '{query}'")
        if results:
            found_names = [r.name for r in results]
            print(f"  Found: {', '.join(found_names)}")

            # Check if expected types are in results
            matches = sum(1 for exp in expected_types if any(exp in name for name in found_names))
            if matches > 0:
                print(f"  [OK] Semantic match successful!")
        else:
            print("  No results")
        print()

    # ========================================
    # Example 4: Execute discovered agents
    # ========================================
    print("Example 4: Discover and Execute")
    print("-" * 70)

    query = "check if there's anything wrong with my code"
    print(f"Query: '{query}'")

    results = await mesh.discover(query, limit=1, min_similarity=0.05)
    if results:
        agent_info = results[0]
        print(f"  Best match: {agent_info.name}")

        # Execute the agent
        result = await mesh.execute(agent_info.id, "def foo(): pass")
        print(f"  Result: {result}")
    else:
        print("  No agent found")

    print()

    # ========================================
    # Example 5: Filters with semantic search
    # ========================================
    print("Example 5: Combining Filters with Semantic Search")
    print("-" * 70)

    # Discover NLP-related agents
    results = await mesh.discover(
        "process natural language",
        limit=5,
        min_similarity=0.05,
        filters={"capabilities": ["nlp"]},
    )

    print(f"Query: 'process natural language' with filter: capabilities=['nlp']")
    print(f"  Found {len(results)} results:")
    for agent_info in results:
        caps = [cap.name for cap in agent_info.capabilities]
        print(f"    - {agent_info.name}: {caps}")

    print()
    print("=" * 70)
    print("Semantic search example completed!")
    print()
    print("Try it with LocalEmbedder for better quality:")
    print("  pip install capabilitymesh[embeddings]")
    print("  mesh = Mesh()  # Auto-selects LocalEmbedder")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
