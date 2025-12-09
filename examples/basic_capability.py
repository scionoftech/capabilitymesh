"""
Basic example demonstrating CapabilityMesh capability definition and usage.

This example shows how to:
1. Create agent identities
2. Define capabilities
3. Work with capability versions
4. Use semantic metadata for discovery
"""

from datetime import datetime
from capabilitymesh import Capability, CapabilityVersion, CapabilityInputOutput, AgentIdentity, AgentAddress
from capabilitymesh.core.types import CapabilityType, AgentType, IOFormat
from capabilitymesh.schemas.capability import UnstructuredCapability, StructuredCapability


def create_sample_agent_identity() -> AgentIdentity:
    """Create a sample agent identity."""
    # In production, use proper key generation
    public_key = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1234567890
-----END PUBLIC KEY-----"""

    address = AgentAddress(protocol="http", host="localhost", port=8000)
    did = AgentIdentity.generate_did(public_key)

    identity = AgentIdentity(
        did=did,
        name="TranslationAgent",
        agent_type=AgentType.LLM,
        description="An AI agent specializing in language translation",
        addresses=[address],
        primary_address=address,
        public_key=public_key,
        created_at=datetime.now(),
        last_seen=datetime.now(),
    )

    return identity


def create_unstructured_capability() -> Capability:
    """Create an unstructured (LLM-based) capability."""
    capability = Capability(
        id="cap-translate-en-fr-001",
        name="translate-en-fr",
        description="Translate text from English to French using advanced LLM",
        version=CapabilityVersion(major=1, minor=0, patch=0),
        capability_type=CapabilityType.UNSTRUCTURED,
        agent_type=AgentType.LLM,
        inputs=[
            CapabilityInputOutput(
                format=IOFormat.TEXT,
                description="English text to translate",
                examples=["Hello, how are you?", "Good morning!"],
            )
        ],
        outputs=[
            CapabilityInputOutput(
                format=IOFormat.TEXT,
                description="French translation",
                examples=["Bonjour, comment allez-vous?", "Bonjour!"],
            )
        ],
        unstructured_spec=UnstructuredCapability(
            prompt_template="Translate the following English text to French: {text}",
            system_prompt="You are a professional translator specializing in English to French translation.",
            model_info={"model": "gpt-4", "provider": "openai"},
            context_window=8000,
        ),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Add semantic metadata for better discovery
    capability.semantic.tags = ["translation", "nlp", "english", "french", "language"]
    capability.semantic.categories = ["language", "translation", "text-processing"]

    # Add constraints
    capability.constraints.timeout_seconds = 30
    capability.constraints.rate_limit = {"requests_per_minute": 60}
    capability.constraints.cost = {"currency": "USD", "per_request": 0.02}

    return capability


def create_structured_capability() -> Capability:
    """Create a structured (API-based) capability."""
    capability = Capability(
        id="cap-sentiment-analysis-001",
        name="sentiment-analysis",
        description="Analyze sentiment of text (positive, negative, neutral)",
        version=CapabilityVersion(major=2, minor=1, patch=0),
        capability_type=CapabilityType.STRUCTURED,
        agent_type=AgentType.SOFTWARE,
        inputs=[
            CapabilityInputOutput(
                format=IOFormat.JSON,
                json_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                description="Text to analyze",
                examples=[{"text": "I love this product!"}],
            )
        ],
        outputs=[
            CapabilityInputOutput(
                format=IOFormat.JSON,
                json_schema={
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["sentiment", "confidence"],
                },
                description="Sentiment analysis result",
                examples=[{"sentiment": "positive", "confidence": 0.95}],
            )
        ],
        structured_spec=StructuredCapability(
            endpoint="/api/v1/sentiment",
            method="POST",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        ),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Add semantic metadata
    capability.semantic.tags = ["sentiment", "analysis", "nlp", "text-processing"]
    capability.semantic.categories = ["nlp", "analysis"]

    return capability


def main():
    """Run the basic capability example."""
    print("=" * 60)
    print("CapabilityMesh Basic Capability Example")
    print("=" * 60)

    # Create agent identity
    print("\n1. Creating Agent Identity...")
    agent = create_sample_agent_identity()
    print(f"   DID: {agent.did}")
    print(f"   Name: {agent.name}")
    print(f"   Type: {agent.agent_type}")
    print(f"   Address: {agent.primary_address.to_uri()}")

    # Create unstructured capability
    print("\n2. Creating Unstructured (LLM) Capability...")
    llm_cap = create_unstructured_capability()
    print(f"   ID: {llm_cap.id}")
    print(f"   Name: {llm_cap.name}")
    print(f"   Version: {llm_cap.version.semver}")
    print(f"   Type: {llm_cap.capability_type}")
    print(f"   Tags: {llm_cap.semantic.tags}")
    print(f"   Cost: ${llm_cap.constraints.cost['per_request']} per request")

    # Create structured capability
    print("\n3. Creating Structured (API) Capability...")
    api_cap = create_structured_capability()
    print(f"   ID: {api_cap.id}")
    print(f"   Name: {api_cap.name}")
    print(f"   Version: {api_cap.version.semver}")
    print(f"   Endpoint: {api_cap.structured_spec.endpoint}")
    print(f"   Method: {api_cap.structured_spec.method}")

    # Demonstrate version compatibility
    print("\n4. Testing Version Compatibility...")
    v1 = CapabilityVersion(major=1, minor=0, patch=0)
    v2 = CapabilityVersion(major=1, minor=5, patch=2)
    v3 = CapabilityVersion(major=2, minor=0, patch=0)

    print(f"   v{v1} compatible with v{v2}? {v1.is_compatible(v2)}")
    print(f"   v{v1} compatible with v{v3}? {v1.is_compatible(v3)}")
    print(f"   v{v2} compatible with v{v3}? {v2.is_compatible(v3)}")

    # Get capability summaries
    print("\n5. Capability Summaries...")
    print("\n   LLM Capability:")
    for key, value in llm_cap.to_summary().items():
        print(f"     {key}: {value}")

    print("\n   API Capability:")
    for key, value in api_cap.to_summary().items():
        print(f"     {key}: {value}")

    # Validate input data
    print("\n6. Validating Input Data...")
    text_input = llm_cap.inputs[0]
    print(f"   Valid text input: {text_input.validate_data('Hello world')}")
    print(f"   Invalid text input: {text_input.validate_data(12345)}")

    json_input = api_cap.inputs[0]
    print(f"   Valid JSON input: {json_input.validate_data({'text': 'test'})}")
    print(f"   Invalid JSON input: {json_input.validate_data('not a dict')}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
