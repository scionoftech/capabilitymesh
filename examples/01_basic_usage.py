"""Example 1: Basic Usage - Registration, Discovery, and Execution

This example demonstrates the core features of CapabilityMesh:
- Registering agents with capabilities
- Discovering agents by capability
- Executing agent tasks
- Using decorators for simple registration

Prerequisites:
    pip install capabilitymesh
    OR
    Set PYTHONPATH to project root
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from capabilitymesh import Mesh, Capability


async def main():
    print("=" * 60)
    print("CapabilityMesh - Basic Usage Example")
    print("=" * 60)

    # Initialize the mesh
    mesh = Mesh()

    # Method 1: Register a function directly
    print("\n1. Registering agents...")

    def summarize_text(text: str) -> str:
        """Summarize the given text."""
        # Simple summarization (in real use, this would call an LLM)
        sentences = text.split('.')
        return sentences[0] + "." if sentences else text

    await mesh.register(
        agent=summarize_text,
        name="text-summarizer",
        capabilities=["text-summarization", "nlp"],
    )
    print("[OK] Registered: text-summarizer")

    # Method 2: Register a regular function explicitly
    def translate_text(text: str, target_lang: str = "es") -> str:
        """Translate text to target language."""
        # Simple mock translation
        return f"[Translated to {target_lang}]: {text}"

    await mesh.register(
        agent=translate_text,
        name="translator",
        capabilities=["translation", "nlp", "language"],
    )
    print("[OK] Registered: translator")

    # Method 3: Register async function
    async def analyze_sentiment(text: str) -> dict:
        """Analyze sentiment of text."""
        # Mock sentiment analysis
        await asyncio.sleep(0.1)  # Simulate async processing
        return {
            "text": text,
            "sentiment": "positive",
            "confidence": 0.95,
        }

    await mesh.register(
        agent=analyze_sentiment,
        name="sentiment-analyzer",
        capabilities=["sentiment-analysis", "nlp"],
    )
    print("[OK] Registered: sentiment-analyzer")

    # Method 4: Register with Capability objects
    capability = Capability.create_simple(
        name="data-extraction",
        description="Extract structured data from text",
        tags=["nlp", "extraction", "structured-data"],
    )

    def extract_data(text: str) -> dict:
        """Extract key-value pairs from text."""
        # Mock data extraction
        return {"entities": ["CapabilityMesh", "Python"], "count": 2}

    await mesh.register(
        agent=extract_data,
        name="data-extractor",
        capabilities=[capability],
    )
    print("[OK] Registered: data-extractor")

    # List all registered agents
    print("\n2. Listing all agents...")
    agents = await mesh.list_agents()
    print(f"Total agents registered: {len(agents)}")
    for agent in agents:
        caps = ", ".join([cap.name for cap in agent.capabilities])
        print(f"  - {agent.name}: [{caps}]")

    # Discover agents by capability
    print("\n3. Discovering agents...")

    # Discover NLP agents
    nlp_agents = await mesh.discover("nlp", limit=5)
    print(f"\nNLP agents found: {len(nlp_agents)}")
    for agent in nlp_agents:
        print(f"  - {agent.name}")

    # Discover translation agents
    translation_agents = await mesh.discover("translation", limit=2)
    print(f"\nTranslation agents found: {len(translation_agents)}")
    for agent in translation_agents:
        print(f"  - {agent.name}")

    # Execute agents
    print("\n4. Executing agent tasks...")

    # Execute text summarization
    if nlp_agents:
        summarizer_id = None
        for agent in nlp_agents:
            if "text-summarization" in [cap.name for cap in agent.capabilities]:
                summarizer_id = agent.id
                break

        if summarizer_id:
            text = "CapabilityMesh is a powerful framework. It enables multi-agent systems. It supports various frameworks."
            result = await mesh.execute(summarizer_id, text)
            print(f"\nSummarization result:")
            print(f"  Input: {text}")
            print(f"  Output: {result}")

    # Execute translation
    if translation_agents:
        translator_id = translation_agents[0].id
        result = await mesh.execute(
            translator_id,
            "Hello, world!",
            target_lang="fr"
        )
        print(f"\nTranslation result:")
        print(f"  Output: {result}")

    # Execute sentiment analysis (async)
    sentiment_agents = await mesh.discover("sentiment", limit=1)
    if sentiment_agents:
        result = await mesh.execute(
            sentiment_agents[0].id,
            "I love using CapabilityMesh!"
        )
        print(f"\nSentiment analysis result:")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Confidence: {result['confidence']}")

    # Get native agent function
    print("\n5. Accessing native functions...")
    if translation_agents:
        native_func = mesh.get_native(translation_agents[0].id)
        if native_func:
            result = native_func("Direct call!", target_lang="de")
            print(f"  Direct function call: {result}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
