"""Example 5: Advanced Capabilities

This example demonstrates advanced capability features:
- Structured capabilities with input/output schemas
- Capability versioning and compatibility
- Capability constraints (performance, cost, SLA)
- Semantic metadata (tags, categories, domains)
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    AgentType,
)


async def main():
    print("=" * 60)
    print("CapabilityMesh - Advanced Capabilities Example")
    print("=" * 60)

    mesh = Mesh()

    # 1. Structured Capability with Input/Output Schema
    print("\n1. Structured capability with I/O schema...")

    input_spec = CapabilityInputOutput(
        format=IOFormat.JSON,
        description="Text to analyze",
        json_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "language": {"type": "string", "default": "en"},
            },
            "required": ["text"],
        },
    )

    output_spec = CapabilityInputOutput(
        format=IOFormat.JSON,
        description="Sentiment analysis result",
        json_schema={
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "emotions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["sentiment", "confidence"],
        },
    )

    import uuid
    from datetime import datetime
    from capabilitymesh.schemas.capability import StructuredCapability

    structured_cap = Capability(
        id=str(uuid.uuid4()),
        name="sentiment-analysis-v2",
        description="Advanced sentiment analysis with emotion detection",
        capability_type=CapabilityType.STRUCTURED,
        agent_type=AgentType.SOFTWARE,
        version=CapabilityVersion.from_string("2.1.0"),
        inputs=[input_spec],
        outputs=[output_spec],
        structured_spec=StructuredCapability(
            endpoint="local",
            method="CALL",
            input_schema=input_spec.json_schema or {},
            output_schema=output_spec.json_schema or {},
        ),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
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

    print("[OK] Registered structured capability")
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

    print(f"[OK] Registered {len(versions)} API versions")

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
        max_concurrent=1000,
        rate_limit={"requests_per_minute": 1000},
        timeout_seconds=1,
        cost={"currency": "USD", "per_request": 0.001},
    )

    from capabilitymesh.schemas.capability import UnstructuredCapability

    fast_cap = Capability(
        id=str(uuid.uuid4()),
        name="fast-translation",
        description="Ultra-fast translation with 99.9% uptime",
        capability_type=CapabilityType.UNSTRUCTURED,
        agent_type=AgentType.SOFTWARE,
        version=CapabilityVersion(major=1, minor=0, patch=0),
        inputs=[CapabilityInputOutput(format=IOFormat.TEXT, description="Input text")],
        outputs=[CapabilityInputOutput(format=IOFormat.TEXT, description="Translated text")],
        unstructured_spec=UnstructuredCapability(),
        constraints=fast_constraints,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    # Budget service with higher latency
    budget_constraints = CapabilityConstraints(
        max_concurrent=100,
        rate_limit={"requests_per_minute": 100},
        timeout_seconds=5,
        cost={"currency": "USD", "per_request": 0.0001},
    )

    budget_cap = Capability(
        id=str(uuid.uuid4()),
        name="budget-translation",
        description="Cost-effective translation service",
        capability_type=CapabilityType.UNSTRUCTURED,
        agent_type=AgentType.SOFTWARE,
        version=CapabilityVersion(major=1, minor=0, patch=0),
        inputs=[CapabilityInputOutput(format=IOFormat.TEXT, description="Input text")],
        outputs=[CapabilityInputOutput(format=IOFormat.TEXT, description="Translated text")],
        unstructured_spec=UnstructuredCapability(),
        constraints=budget_constraints,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
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

    print("[OK] Registered services with constraints")
    print(f"\n  Fast Service:")
    print(f"    Timeout: {fast_constraints.timeout_seconds}s")
    print(f"    Max cost: ${fast_constraints.cost.get('per_request')}")
    print(f"    Max concurrent: {fast_constraints.max_concurrent}")
    print(f"    Rate limit: {fast_constraints.rate_limit.get('requests_per_minute')}/min")
    print(f"\n  Budget Service:")
    print(f"    Timeout: {budget_constraints.timeout_seconds}s")
    print(f"    Max cost: ${budget_constraints.cost.get('per_request')}")
    print(f"    Max concurrent: {budget_constraints.max_concurrent}")
    print(f"    Rate limit: {budget_constraints.rate_limit.get('requests_per_minute')}/min")

    # 4. Semantic Metadata
    print("\n4. Semantic metadata (tags, categories, domains)...")

    semantic_metadata = SemanticMetadata(
        tags=["nlp", "text-processing", "ai", "machine-learning", "healthcare", "medical"],
        categories=["Natural Language Processing", "Text Analysis", "Healthcare"],
    )

    medical_cap = Capability(
        id=str(uuid.uuid4()),
        name="medical-text-analysis",
        description="Analyze medical and clinical text documents",
        capability_type=CapabilityType.UNSTRUCTURED,
        agent_type=AgentType.SOFTWARE,
        version=CapabilityVersion(major=1, minor=0, patch=0),
        inputs=[CapabilityInputOutput(format=IOFormat.TEXT, description="Medical text input")],
        outputs=[CapabilityInputOutput(format=IOFormat.JSON, description="Analysis results")],
        unstructured_spec=UnstructuredCapability(),
        semantic=semantic_metadata,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
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

    print("[OK] Registered capability with rich metadata")
    print(f"  Tags: {', '.join(semantic_metadata.tags)}")
    print(f"  Categories: {', '.join(semantic_metadata.categories)}")

    # 5. Capability Summary
    print("\n5. Capability summaries...")

    agents = await mesh.list_agents()
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

    print("[OK] Registered deprecated capability")
    print(f"  Deprecated: {old_cap.is_deprecated()}")
    print(f"  Notice: {old_cap.deprecation_notice}")

    # 7. Hybrid Capabilities
    print("\n7. Hybrid capabilities (structured + unstructured)...")

    hybrid_input = CapabilityInputOutput(
        format=IOFormat.JSON,
        description="Accepts both JSON and plain text",
        json_schema={"type": "object"},
    )

    hybrid_output = CapabilityInputOutput(
        format=IOFormat.JSON,
        description="Returns structured data or plain text",
        json_schema={"type": "object"},
    )

    hybrid_cap = Capability(
        id=str(uuid.uuid4()),
        name="hybrid-processor",
        description="Flexible processor that handles multiple formats",
        capability_type=CapabilityType.HYBRID,
        agent_type=AgentType.SOFTWARE,
        version=CapabilityVersion(major=1, minor=0, patch=0),
        inputs=[hybrid_input],
        outputs=[hybrid_output],
        structured_spec=StructuredCapability(
            endpoint="local",
            method="CALL",
            input_schema=hybrid_input.json_schema or {},
            output_schema=hybrid_output.json_schema or {},
        ),
        unstructured_spec=UnstructuredCapability(),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
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

    print("[OK] Registered hybrid capability")
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
