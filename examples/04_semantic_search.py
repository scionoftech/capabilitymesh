"""Example 4: Semantic Search

This example demonstrates semantic discovery using embeddings:
- Keyword-based embeddings (default)
- Semantic similarity matching
- Custom embedders
- Search with different similarity thresholds
"""

import asyncio
from capabilitymesh import Mesh, Capability, KeywordEmbedder


async def main():
    print("=" * 60)
    print("CapabilityMesh - Semantic Search Example")
    print("=" * 60)

    # Initialize mesh (uses KeywordEmbedder by default)
    mesh = Mesh()

    # Register agents with detailed descriptions
    print("\n1. Registering agents with rich descriptions...")

    agents_data = [
        {
            "name": "text-summarizer",
            "description": "Summarizes long articles and documents into concise summaries",
            "capabilities": ["summarization", "text-processing", "nlp"],
        },
        {
            "name": "translation-service",
            "description": "Translates text between multiple languages including English, Spanish, French",
            "capabilities": ["translation", "language-processing", "nlp"],
        },
        {
            "name": "sentiment-analyzer",
            "description": "Analyzes sentiment and emotion in text, identifies positive, negative, neutral tones",
            "capabilities": ["sentiment-analysis", "emotion-detection", "nlp"],
        },
        {
            "name": "entity-extractor",
            "description": "Extracts named entities like people, organizations, locations from text",
            "capabilities": ["entity-extraction", "ner", "nlp"],
        },
        {
            "name": "image-classifier",
            "description": "Classifies images into categories using deep learning models",
            "capabilities": ["image-classification", "computer-vision", "deep-learning"],
        },
        {
            "name": "object-detector",
            "description": "Detects and localizes objects in images with bounding boxes",
            "capabilities": ["object-detection", "computer-vision", "deep-learning"],
        },
        {
            "name": "speech-recognizer",
            "description": "Converts speech audio to text using automatic speech recognition",
            "capabilities": ["speech-recognition", "audio-processing", "asr"],
        },
        {
            "name": "text-to-speech",
            "description": "Converts text to natural sounding speech audio",
            "capabilities": ["text-to-speech", "audio-generation", "tts"],
        },
        {
            "name": "code-generator",
            "description": "Generates Python, JavaScript, and other programming language code",
            "capabilities": ["code-generation", "programming", "ai-coding"],
        },
        {
            "name": "data-analyzer",
            "description": "Analyzes datasets, performs statistical analysis, generates insights",
            "capabilities": ["data-analysis", "statistics", "insights"],
        },
    ]

    for agent_data in agents_data:
        def dummy_func(input_data):
            return f"Processed by {agent_data['name']}"

        capability = Capability.create_simple(
            name=agent_data["capabilities"][0],
            description=agent_data["description"],
            tags=agent_data["capabilities"],
        )

        await mesh.register(
            agent=dummy_func,
            name=agent_data["name"],
            capabilities=[capability],
        )

    print(f"✓ Registered {len(agents_data)} agents")

    # Semantic search examples
    print("\n2. Semantic search examples...")

    search_queries = [
        "I need to understand the mood of customer reviews",
        "Help me convert documents to different languages",
        "Find objects and things in my photos",
        "Make a computer program automatically",
        "Turn written text into spoken words",
        "Identify people and places mentioned in articles",
    ]

    for query in search_queries:
        print(f"\n  Query: '{query}'")
        agents = await mesh.discover(query, limit=3)

        if agents:
            print(f"  Top match: {agents[0].name}")
            print(f"  Description: {agents[0].capabilities[0].description[:60]}...")

    # Detailed similarity matching
    print("\n3. Detailed similarity matching...")

    query = "analyze customer feedback sentiment"
    print(f"\n  Query: '{query}'")

    agents = await mesh.discover(query, limit=5, min_similarity=0.001)
    print(f"  Found {len(agents)} agents")

    for i, agent in enumerate(agents, 1):
        # Get capability descriptions
        cap_desc = agent.capabilities[0].description if agent.capabilities else "No description"
        cap_tags = ", ".join(agent.capabilities[0].semantic.tags) if agent.capabilities and agent.capabilities[0].semantic else "No tags"

        print(f"\n  {i}. {agent.name}")
        print(f"     Description: {cap_desc[:70]}...")
        print(f"     Tags: {cap_tags}")

    # Different similarity thresholds
    print("\n4. Testing different similarity thresholds...")

    query = "natural language processing"
    print(f"\n  Query: '{query}'")

    for threshold in [0.0, 0.1, 0.2, 0.3]:
        agents = await mesh.discover(query, limit=10, min_similarity=threshold)
        print(f"  Threshold {threshold:.1f}: {len(agents)} agents found")

    # Category-based discovery
    print("\n5. Category-based discovery...")

    categories = {
        "NLP": ["text", "nlp", "language", "sentiment", "translation"],
        "Computer Vision": ["image", "vision", "object", "detection"],
        "Audio": ["speech", "audio", "sound", "voice"],
        "Code": ["code", "programming", "generation"],
    }

    for category, keywords in categories.items():
        query = " ".join(keywords)
        agents = await mesh.discover(query, limit=5)
        print(f"\n  {category}:")
        for agent in agents:
            print(f"    - {agent.name}")

    # Demonstrate embedder functionality
    print("\n6. Understanding the embedder...")

    embedder = mesh.embedder
    print(f"  Embedder type: {embedder.model_name}")
    print(f"  Embedding dimension: {embedder.dimension}")

    # Generate embeddings for sample texts
    texts = [
        "sentiment analysis",
        "emotion detection",
        "image classification",
    ]

    print("\n  Sample embeddings:")
    for text in texts:
        embedding = await embedder.embed(text)
        # Show first 5 dimensions
        print(f"    '{text}':")
        print(f"      First 5 dims: [{', '.join(f'{x:.4f}' for x in embedding[:5])}...]")
        print(f"      Length: {len(embedding)}")

    # Similarity calculation
    print("\n7. Calculating semantic similarity...")

    from capabilitymesh.embeddings import cosine_similarity

    text_pairs = [
        ("sentiment analysis", "emotion detection"),
        ("sentiment analysis", "image classification"),
        ("translate text", "language translation"),
        ("image classifier", "speech recognition"),
    ]

    for text1, text2 in text_pairs:
        emb1 = await embedder.embed(text1)
        emb2 = await embedder.embed(text2)
        similarity = cosine_similarity(emb1, emb2)
        print(f"\n  '{text1}' <-> '{text2}'")
        print(f"    Similarity: {similarity:.4f}")

    # Custom search with filters
    print("\n8. Search with capability filters...")

    # Search for NLP agents only
    query = "process text"
    agents = await mesh.storage.search(query, limit=10, filters={"capabilities": ["nlp"]})

    print(f"\n  Query: '{query}' (filtered for 'nlp' capability)")
    print(f"  Found {len(agents)} agents:")
    for agent in agents:
        print(f"    - {agent.identity.name}")

    # Tips for semantic search
    print("\n" + "=" * 60)
    print("Semantic Search Tips")
    print("=" * 60)
    print("\n1. Use descriptive capability descriptions")
    print("   - Include key terms and synonyms")
    print("   - Describe what the agent does, not how")
    print("\n2. Add relevant tags to capabilities")
    print("   - Use domain-specific terminology")
    print("   - Include common variations")
    print("\n3. Adjust similarity threshold based on needs")
    print("   - Lower threshold: More results, less precise")
    print("   - Higher threshold: Fewer results, more precise")
    print("\n4. Combine semantic search with filters")
    print("   - Filter by capability names or tags")
    print("   - Use trust levels for quality filtering")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
