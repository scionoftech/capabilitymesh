"""Example 6: Multi-Agent Workflow

This example demonstrates coordinating multiple agents:
- Sequential workflow (pipeline)
- Parallel workflow (fan-out/fan-in)
- Conditional routing
- Error handling and fallbacks
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from typing import List, Dict
from capabilitymesh import Mesh, Capability


async def main():
    print("=" * 60)
    print("CapabilityMesh - Multi-Agent Workflow Example")
    print("=" * 60)

    mesh = Mesh()

    # Register agents for a document processing workflow
    print("\n1. Registering workflow agents...")

    # Step 1: Text extraction
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF."""
        return f"Extracted text from {pdf_path}: This is a sample document about AI and machine learning."

    await mesh.register(extract_text_from_pdf, name="pdf-extractor", capabilities=["text-extraction", "pdf"])

    # Step 2: Text preprocessing
    def preprocess_text(text: str) -> str:
        """Clean and normalize text."""
        cleaned = text.strip().lower()
        return f"[Preprocessed] {cleaned}"

    await mesh.register(preprocess_text, name="text-preprocessor", capabilities=["preprocessing", "text"])

    # Step 3: Entity extraction
    def extract_entities(text: str) -> List[str]:
        """Extract named entities."""
        return ["AI", "machine learning", "sample document"]

    await mesh.register(extract_entities, name="entity-extractor", capabilities=["entity-extraction", "ner"])

    # Step 4: Sentiment analysis
    def analyze_sentiment(text: str) -> str:
        """Analyze sentiment."""
        return "positive"

    await mesh.register(analyze_sentiment, name="sentiment-analyzer", capabilities=["sentiment-analysis"])

    # Step 5: Summarization
    def summarize_text(text: str) -> str:
        """Generate summary."""
        return "This document discusses AI and machine learning concepts."

    await mesh.register(summarize_text, name="summarizer", capabilities=["summarization"])

    # Step 6: Translation
    def translate_text(text: str, target_lang: str = "es") -> str:
        """Translate text."""
        return f"[Translated to {target_lang}] {text}"

    await mesh.register(translate_text, name="translator", capabilities=["translation"])

    # Step 7: Report generation
    def generate_report(data: Dict) -> str:
        """Generate final report."""
        return f"""
Document Analysis Report
========================
Entities: {', '.join(data['entities'])}
Sentiment: {data['sentiment']}
Summary: {data['summary']}
Translation: {data['translation']}
"""

    await mesh.register(generate_report, name="report-generator", capabilities=["report-generation"])

    print("[OK] Registered 7 workflow agents")

    # Workflow 1: Sequential Pipeline
    print("\n2. Sequential workflow (pipeline)...")

    async def sequential_workflow(pdf_path: str) -> str:
        """Process document through sequential steps."""
        print(f"\n  Starting pipeline for: {pdf_path}")

        # Step 1: Extract text
        extractors = await mesh.discover("text extraction pdf")
        text = await mesh.execute(extractors[0].id, pdf_path)
        print(f"  [OK] Extracted text")

        # Step 2: Preprocess
        preprocessors = await mesh.discover("preprocessing text")
        processed_text = await mesh.execute(preprocessors[0].id, text)
        print(f"  [OK] Preprocessed text")

        # Step 3: Summarize
        summarizers = await mesh.discover("summarization")
        summary = await mesh.execute(summarizers[0].id, processed_text)
        print(f"  [OK] Generated summary")

        return summary

    result = await sequential_workflow("document.pdf")
    print(f"\n  Final result: {result}")

    # Workflow 2: Parallel Processing (Fan-out/Fan-in)
    print("\n3. Parallel workflow (fan-out/fan-in)...")

    async def parallel_workflow(pdf_path: str) -> Dict:
        """Process document with parallel analysis."""
        print(f"\n  Starting parallel workflow for: {pdf_path}")

        # Step 1: Extract text (sequential)
        extractors = await mesh.discover("text extraction")
        text = await mesh.execute(extractors[0].id, pdf_path)
        print(f"  [OK] Extracted text")

        # Step 2: Parallel processing (fan-out)
        print(f"  → Fanning out to parallel tasks...")

        # Discover all agents first
        entity_agents = await mesh.discover("entity extraction")
        sentiment_agents = await mesh.discover("sentiment analysis")
        summary_agents = await mesh.discover("summarization")
        translation_agents = await mesh.discover("translation")

        # Create helper functions for tasks
        async def run_entity_extraction():
            if entity_agents:
                return await mesh.execute(entity_agents[0].id, text)
            return []

        async def run_sentiment_analysis():
            if sentiment_agents:
                return await mesh.execute(sentiment_agents[0].id, text)
            return "neutral"

        async def run_summarization():
            if summary_agents:
                return await mesh.execute(summary_agents[0].id, text)
            return ""

        async def run_translation():
            if translation_agents:
                return await mesh.execute(translation_agents[0].id, text, target_lang="fr")
            return ""

        # Create tasks
        entity_task = asyncio.create_task(run_entity_extraction())
        sentiment_task = asyncio.create_task(run_sentiment_analysis())
        summary_task = asyncio.create_task(run_summarization())
        translation_task = asyncio.create_task(run_translation())

        # Wait for all tasks to complete (fan-in)
        entities, sentiment, summary, translation = await asyncio.gather(
            entity_task,
            sentiment_task,
            summary_task,
            translation_task,
        )

        print(f"  [OK] Completed parallel tasks")

        # Step 3: Combine results
        result = {
            "entities": entities,
            "sentiment": sentiment,
            "summary": summary,
            "translation": translation,
        }

        # Step 4: Generate report
        report_agents = await mesh.discover("report generation")
        report = await mesh.execute(report_agents[0].id, result)
        print(f"  [OK] Generated final report")

        return report

    report = await parallel_workflow("document.pdf")
    print(f"\n  Report:\n{report}")

    # Workflow 3: Conditional Routing
    print("\n4. Conditional routing...")

    async def conditional_workflow(text: str, language: str) -> str:
        """Route based on conditions."""
        print(f"\n  Processing text in {language}...")

        # Check if translation is needed
        if language != "en":
            print(f"  → Translating to English first...")
            translators = await mesh.discover("translation")
            if translators:
                text = await mesh.execute(translators[0].id, text, target_lang="en")
                print(f"  [OK] Translated to English")

        # Analyze sentiment
        sentiment_agents = await mesh.discover("sentiment")
        sentiment = await mesh.execute(sentiment_agents[0].id, text)
        print(f"  [OK] Sentiment: {sentiment}")

        # Route based on sentiment
        if sentiment == "negative":
            print(f"  → Detected negative sentiment, applying special handling...")
            # Could route to escalation agent
            return f"[ALERT] Negative sentiment detected in: {text}"
        else:
            print(f"  → Positive/neutral sentiment, standard processing...")
            return f"[OK] Processed: {text}"

    result1 = await conditional_workflow("This is great!", "en")
    print(f"  Result: {result1}")

    # Workflow 4: Error Handling and Fallbacks
    print("\n5. Error handling and fallbacks...")

    # Register a unreliable agent
    def unreliable_operation(input_data: str) -> str:
        """Sometimes fails."""
        import random
        if random.random() < 0.7:
            raise ValueError("Service temporarily unavailable")
        return f"Success: {input_data}"

    await mesh.register(unreliable_operation, name="unreliable-service", capabilities=["unreliable"])

    # Register a fallback agent
    def fallback_operation(input_data: str) -> str:
        """Always succeeds."""
        return f"Fallback processed: {input_data}"

    await mesh.register(fallback_operation, name="fallback-service", capabilities=["fallback"])

    async def workflow_with_fallback(input_data: str) -> str:
        """Try primary service, fallback on error."""
        print(f"\n  Attempting primary service...")

        # Try primary service
        try:
            primary_agents = await mesh.discover("unreliable")
            result = await mesh.execute(primary_agents[0].id, input_data)
            print(f"  [OK] Primary service succeeded")
            return result
        except Exception as e:
            print(f"  ⚠ Primary service failed: {e}")
            print(f"  → Trying fallback service...")

            # Fallback to backup service
            fallback_agents = await mesh.discover("fallback")
            result = await mesh.execute(fallback_agents[0].id, input_data)
            print(f"  [OK] Fallback service succeeded")
            return result

    # Try the workflow a few times
    for i in range(3):
        result = await workflow_with_fallback(f"task-{i}")
        print(f"  Result: {result}")

    # Workflow 5: Dynamic Agent Selection
    print("\n6. Dynamic agent selection based on criteria...")

    async def select_best_agent(capability: str, min_trust_level=None) -> str:
        """Select best available agent."""
        print(f"\n  Finding best agent for: {capability}")

        # Discover agents
        agents = await mesh.discover(capability, limit=5)

        if not agents:
            return "No agents found"

        # Could add additional selection criteria here:
        # - Trust scores
        # - Performance metrics
        # - Cost
        # - Availability

        selected = agents[0]
        print(f"  [OK] Selected: {selected.name}")

        # Execute
        result = await mesh.execute(selected.id, "test input")
        return result

    await select_best_agent("summarization")

    # Summary
    print("\n" + "=" * 60)
    print("Workflow Patterns Summary")
    print("=" * 60)
    print("\n1. Sequential Pipeline:")
    print("   Agent A → Agent B → Agent C")
    print("   Best for: Step-by-step processing")
    print("\n2. Parallel Processing:")
    print("   Agent A → [Agent B, Agent C, Agent D] → Agent E")
    print("   Best for: Independent parallel tasks")
    print("\n3. Conditional Routing:")
    print("   Agent A → if condition: Agent B else: Agent C")
    print("   Best for: Dynamic decision-making")
    print("\n4. Error Handling:")
    print("   try: Agent A except: Agent B (fallback)")
    print("   Best for: Reliability and resilience")
    print("\n5. Dynamic Selection:")
    print("   Select best agent based on trust, cost, performance")
    print("   Best for: Optimization and quality")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
