"""Example 5: Advanced Capabilities

This example demonstrates advanced capability features:
- Structured capabilities with input/output schemas
- Capability versioning and compatibility
- Capability constraints (performance, cost, SLA)
- Semantic metadata (tags, categories, domains)
"""

import asyncio
from capabilitymesh import (
    Mesh,
    Capability,
    CapabilityType,
    CapabilityVersion,
    CapabilityInputOutput,
    CapabilityConstraints,
    SemanticMetadata,
    IOFormat,
)


async def main():
    print("=" * 60)
    print("CapabilityMesh - Advanced Capabilities Example")
    print("=" * 60)

    mesh = Mesh()

    # 1. Structured Capability with Input/Output Schema
    print("\n1. Structured capability with I/O schema...")

    input_spec = CapabilityInputOutput(
        name="text_input",
        description="Text to analyze",
        format=IOFormat.TEXT,
        schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "language": {"type": "string", "default": "en"},
            },
            "required": ["text"],
        },
    )

    output_spec = CapabilityInputOutput(
        name="analysis_result",
        description="Sentiment analysis result",
        format=IOFormat.JSON,
        schema={
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "emotions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["sentiment", "confidence"],
        },
    )

    structured_cap = Capability(
        name="sentiment-analysis-v2",
        description="Advanced sentiment analysis with emotion detection",
        capability_type=CapabilityType.STRUCTURED,
        version=CapabilityVersion.from_string("2.1.0"),
        structured={"input": input_spec, "output": output_spec},
    )

    def analyze_sentiment_v2(data: dict) -> dict:
        """Analyze sentiment with structured I/O."""
        text = data.get("text", "")
        return {
            "sentiment": "positive",
            "confidence": 0.92,
            "emotions": ["joy", "excitement"],
        }

    await mesh.register(
        agent=analyze_sentiment_v2,
        name="sentiment-analyzer-v2",
        capabilities=[structured_cap],
    )

    print("✓ Registered structured capability")
    print(f"  Input format: {input_spec.format}")
    print(f"  Output format: {output_spec.format}")
    print(f"  Input schema: {input_spec.schema}")

    # 2. Capability Versioning
    print("\n2. Capability versioning...")

    versions = ["1.0.0", "1.5.2", "2.0.0", "2.1.0"]

    for ver_str in versions:
        version = CapabilityVersion.from_string(ver_str)

        cap = Capability.create_simple(
            name=f"api-{ver_str.replace('.', '-')}",
            description=f"API version {ver_str}",
            tags=["api"],
        )
        cap.version = version

        def api_func(input_data):
            return f"Processed by API v{ver_str}"

        await mesh.register(
            agent=api_func,
            name=f"api-service-{ver_str}",
            capabilities=[cap],
        )

    print(f"✓ Registered {len(versions)} API versions")

    # Check version compatibility
    print("\n  Version compatibility checks:")
    v1 = CapabilityVersion.from_string("1.5.0")
    v2 = CapabilityVersion.from_string("1.6.0")
    v3 = CapabilityVersion.from_string("2.0.0")

    print(f"    v1.5.0 compatible with v1.6.0? {v1.is_compatible(v2)}")
    print(f"    v1.5.0 compatible with v2.0.0? {v1.is_compatible(v3)}")
    print(f"    v1.5.0 > v1.6.0? {v1 > v2}")
    print(f"    v2.0.0 > v1.5.0? {v3 > v1}")

    # 3. Capability Constraints
    print("\n3. Capability constraints (performance, cost, SLA)...")

    # High-performance, low-latency service
    fast_constraints = CapabilityConstraints(
        max_response_time_ms=100,
        max_cost_per_call=0.001,
        min_availability=0.999,
        rate_limit_per_minute=1000,
    )

    fast_cap = Capability(
        name="fast-translation",
        description="Ultra-fast translation with 99.9% uptime",
        capability_type=CapabilityType.UNSTRUCTURED,
        constraints=fast_constraints,
    )

    # Budget service with higher latency
    budget_constraints = CapabilityConstraints(
        max_response_time_ms=5000,
        max_cost_per_call=0.0001,
        min_availability=0.95,
        rate_limit_per_minute=100,
    )

    budget_cap = Capability(
        name="budget-translation",
        description="Cost-effective translation service",
        capability_type=CapabilityType.UNSTRUCTURED,
        constraints=budget_constraints,
    )

    def fast_translate(text: str) -> str:
        return f"[Fast] Translated: {text}"

    def budget_translate(text: str) -> str:
        return f"[Budget] Translated: {text}"

    await mesh.register(
        agent=fast_translate,
        name="fast-translator",
        capabilities=[fast_cap],
    )

    await mesh.register(
        agent=budget_translate,
        name="budget-translator",
        capabilities=[budget_cap],
    )

    print("✓ Registered services with constraints")
    print(f"\n  Fast Service:")
    print(f"    Max latency: {fast_constraints.max_response_time_ms}ms")
    print(f"    Max cost: ${fast_constraints.max_cost_per_call}")
    print(f"    Availability: {fast_constraints.min_availability * 100}%")
    print(f"    Rate limit: {fast_constraints.rate_limit_per_minute}/min")
    print(f"\n  Budget Service:")
    print(f"    Max latency: {budget_constraints.max_response_time_ms}ms")
    print(f"    Max cost: ${budget_constraints.max_cost_per_call}")
    print(f"    Availability: {budget_constraints.min_availability * 100}%")
    print(f"    Rate limit: {budget_constraints.rate_limit_per_minute}/min")

    # 4. Semantic Metadata
    print("\n4. Semantic metadata (tags, categories, domains)...")

    semantic_metadata = SemanticMetadata(
        tags=["nlp", "text-processing", "ai", "machine-learning"],
        categories=["Natural Language Processing", "Text Analysis"],
        domains=["healthcare", "finance", "legal"],
        keywords=["medical", "clinical", "diagnosis", "patient"],
    )

    medical_cap = Capability(
        name="medical-text-analysis",
        description="Analyze medical and clinical text documents",
        capability_type=CapabilityType.UNSTRUCTURED,
        semantic=semantic_metadata,
    )

    def analyze_medical_text(text: str) -> dict:
        return {
            "entities": ["diagnosis", "medication"],
            "domain": "healthcare",
        }

    await mesh.register(
        agent=analyze_medical_text,
        name="medical-analyzer",
        capabilities=[medical_cap],
    )

    print("✓ Registered capability with rich metadata")
    print(f"  Tags: {', '.join(semantic_metadata.tags)}")
    print(f"  Categories: {', '.join(semantic_metadata.categories)}")
    print(f"  Domains: {', '.join(semantic_metadata.domains)}")
    print(f"  Keywords: {', '.join(semantic_metadata.keywords)}")

    # 5. Capability Summary
    print("\n5. Capability summaries...")

    agents = await mesh.list()
    print(f"\n  Found {len(agents)} agents with capabilities:")

    for agent in agents[:5]:  # Show first 5
        if agent.capabilities:
            cap = agent.capabilities[0]
            summary = cap.to_summary()
            print(f"\n  {agent.name}:")
            print(f"    {summary}")

    # 6. Deprecated Capabilities
    print("\n6. Marking capabilities as deprecated...")

    old_cap = Capability.create_simple(
        name="old-api",
        description="Legacy API - please use v2",
        tags=["api", "legacy"],
    )
    old_cap.deprecated = True
    old_cap.deprecation_notice = "This API will be removed in v3.0. Use sentiment-analysis-v2 instead."

    def old_api_func(input_data):
        return "Old API response"

    await mesh.register(
        agent=old_api_func,
        name="old-api-service",
        capabilities=[old_cap],
    )

    print("✓ Registered deprecated capability")
    print(f"  Deprecated: {old_cap.is_deprecated}")
    print(f"  Notice: {old_cap.deprecation_notice}")

    # 7. Hybrid Capabilities
    print("\n7. Hybrid capabilities (structured + unstructured)...")

    hybrid_input = CapabilityInputOutput(
        name="flexible_input",
        description="Accepts both JSON and plain text",
        format=IOFormat.JSON,
        schema={"type": "object"},
    )

    hybrid_output = CapabilityInputOutput(
        name="flexible_output",
        description="Returns structured data or plain text",
        format=IOFormat.JSON,
        schema={"type": "object"},
    )

    hybrid_cap = Capability(
        name="hybrid-processor",
        description="Flexible processor that handles multiple formats",
        capability_type=CapabilityType.HYBRID,
        structured={"input": hybrid_input, "output": hybrid_output},
        unstructured={"input_description": "Any text or JSON", "output_description": "Processed result"},
    )

    def hybrid_process(input_data) -> dict:
        if isinstance(input_data, dict):
            return {"type": "json", "processed": True}
        return {"type": "text", "processed": str(input_data)}

    await mesh.register(
        agent=hybrid_process,
        name="hybrid-processor",
        capabilities=[hybrid_cap],
    )

    print("✓ Registered hybrid capability")
    print(f"  Type: {hybrid_cap.capability_type}")
    print(f"  Supports: Both structured (JSON) and unstructured (text) inputs")

    # Summary
    print("\n" + "=" * 60)
    print("Capability Features Summary")
    print("=" * 60)
    print("\n1. Types:")
    print("   - STRUCTURED: Well-defined I/O schemas")
    print("   - UNSTRUCTURED: Flexible text-based I/O")
    print("   - HYBRID: Supports both formats")
    print("\n2. Versioning:")
    print("   - Semantic versioning (major.minor.patch)")
    print("   - Compatibility checking")
    print("   - Version comparison")
    print("\n3. Constraints:")
    print("   - Performance (latency, throughput)")
    print("   - Cost (per-call pricing)")
    print("   - Availability (uptime SLA)")
    print("   - Rate limits")
    print("\n4. Metadata:")
    print("   - Tags (for categorization)")
    print("   - Categories (hierarchical)")
    print("   - Domains (industry-specific)")
    print("   - Keywords (for search)")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
