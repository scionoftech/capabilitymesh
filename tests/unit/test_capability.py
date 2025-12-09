"""Unit tests for capability schema."""

import pytest
from datetime import datetime

from capabilitymesh.schemas.capability import (
    Capability,
    CapabilityVersion,
    CapabilityInputOutput,
    StructuredCapability,
    UnstructuredCapability,
    CapabilityConstraints,
    SemanticMetadata,
)
from capabilitymesh.core.types import CapabilityType, AgentType, IOFormat


class TestCapabilityVersion:
    """Tests for CapabilityVersion class."""

    def test_create_version(self):
        """Test creating a capability version."""
        version = CapabilityVersion(major=1, minor=2, patch=3)

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_semver_property(self):
        """Test semantic version string property."""
        version = CapabilityVersion(major=2, minor=5, patch=10)

        assert version.semver == "2.5.10"

    def test_from_string(self):
        """Test creating version from string."""
        version = CapabilityVersion.from_string("1.2.3")

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_from_string_invalid(self):
        """Test creating version from invalid string."""
        with pytest.raises(ValueError):
            CapabilityVersion.from_string("1.2")

        with pytest.raises(ValueError):
            CapabilityVersion.from_string("invalid")

        with pytest.raises(ValueError):
            CapabilityVersion.from_string("1.2.3.4")

    def test_str_representation(self):
        """Test string representation."""
        version = CapabilityVersion(major=1, minor=0, patch=0)

        assert str(version) == "1.0.0"

    def test_is_compatible(self):
        """Test version compatibility checking."""
        v1 = CapabilityVersion(major=1, minor=2, patch=3)
        v2 = CapabilityVersion(major=1, minor=5, patch=0)
        v3 = CapabilityVersion(major=2, minor=0, patch=0)

        assert v1.is_compatible(v2) is True  # Same major version
        assert v1.is_compatible(v3) is False  # Different major version

    def test_version_comparisons(self):
        """Test version comparison operators."""
        v1 = CapabilityVersion(major=1, minor=0, patch=0)
        v2 = CapabilityVersion(major=1, minor=2, patch=0)
        v3 = CapabilityVersion(major=2, minor=0, patch=0)

        assert v1 < v2
        assert v1 < v3
        assert v2 < v3
        assert v3 > v2
        assert v2 >= v2
        assert v1 <= v2


class TestCapabilityInputOutput:
    """Tests for CapabilityInputOutput class."""

    def test_create_io_spec(self):
        """Test creating an IO specification."""
        io_spec = CapabilityInputOutput(
            format=IOFormat.JSON,
            description="Input data",
            json_schema={"type": "object"},
            examples=[{"key": "value"}],
        )

        assert io_spec.format == IOFormat.JSON
        assert io_spec.description == "Input data"
        assert io_spec.json_schema == {"type": "object"}
        assert len(io_spec.examples) == 1

    def test_validate_data_json(self):
        """Test data validation for JSON format."""
        io_spec = CapabilityInputOutput(
            format=IOFormat.JSON,
            description="JSON input",
        )

        assert io_spec.validate_data({"key": "value"}) is True
        assert io_spec.validate_data("not a dict") is False

    def test_validate_data_text(self):
        """Test data validation for TEXT format."""
        io_spec = CapabilityInputOutput(
            format=IOFormat.TEXT,
            description="Text input",
        )

        assert io_spec.validate_data("Hello world") is True
        assert io_spec.validate_data(123) is False


class TestStructuredCapability:
    """Tests for StructuredCapability class."""

    def test_create_structured_spec(self):
        """Test creating a structured capability specification."""
        spec = StructuredCapability(
            endpoint="/api/v1/translate",
            method="POST",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"translation": {"type": "string"}}},
        )

        assert spec.endpoint == "/api/v1/translate"
        assert spec.method == "POST"
        assert "text" in spec.input_schema["properties"]


class TestUnstructuredCapability:
    """Tests for UnstructuredCapability class."""

    def test_create_unstructured_spec(self):
        """Test creating an unstructured capability specification."""
        spec = UnstructuredCapability(
            prompt_template="Translate the following text to {target_language}: {text}",
            system_prompt="You are a helpful translation assistant",
            model_info={"model": "gpt-4", "provider": "openai"},
            context_window=8000,
        )

        assert "target_language" in spec.prompt_template
        assert spec.context_window == 8000
        assert spec.model_info["model"] == "gpt-4"


class TestCapability:
    """Tests for Capability class."""

    @pytest.fixture
    def sample_structured_capability(self):
        """Sample structured capability for testing."""
        return Capability(
            id="cap-123",
            name="translate-en-fr",
            description="Translate English to French",
            version=CapabilityVersion(major=1, minor=0, patch=0),
            capability_type=CapabilityType.STRUCTURED,
            agent_type=AgentType.SOFTWARE,
            inputs=[
                CapabilityInputOutput(
                    format=IOFormat.JSON,
                    description="English text to translate",
                    examples=[{"text": "Hello"}],
                )
            ],
            outputs=[
                CapabilityInputOutput(
                    format=IOFormat.JSON,
                    description="French translation",
                    examples=[{"text": "Bonjour"}],
                )
            ],
            structured_spec=StructuredCapability(
                endpoint="/translate",
                method="POST",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            ),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.fixture
    def sample_unstructured_capability(self):
        """Sample unstructured capability for testing."""
        return Capability(
            id="cap-456",
            name="summarize-text",
            description="Summarize long text",
            version=CapabilityVersion(major=1, minor=0, patch=0),
            capability_type=CapabilityType.UNSTRUCTURED,
            agent_type=AgentType.LLM,
            inputs=[
                CapabilityInputOutput(
                    format=IOFormat.TEXT,
                    description="Text to summarize",
                )
            ],
            outputs=[
                CapabilityInputOutput(
                    format=IOFormat.TEXT,
                    description="Summary",
                )
            ],
            unstructured_spec=UnstructuredCapability(
                prompt_template="Summarize: {text}",
                system_prompt="You are a summarization assistant",
            ),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def test_create_structured_capability(self, sample_structured_capability):
        """Test creating a structured capability."""
        cap = sample_structured_capability

        assert cap.id == "cap-123"
        assert cap.name == "translate-en-fr"
        assert cap.capability_type == CapabilityType.STRUCTURED
        assert cap.structured_spec is not None
        assert cap.unstructured_spec is None

    def test_create_unstructured_capability(self, sample_unstructured_capability):
        """Test creating an unstructured capability."""
        cap = sample_unstructured_capability

        assert cap.id == "cap-456"
        assert cap.capability_type == CapabilityType.UNSTRUCTURED
        assert cap.unstructured_spec is not None
        assert cap.structured_spec is None

    def test_structured_capability_requires_spec(self):
        """Test that structured capabilities require structured_spec."""
        with pytest.raises(ValueError, match="Structured capabilities must have structured_spec"):
            Capability(
                id="cap-invalid",
                name="invalid",
                description="Invalid structured capability",
                version=CapabilityVersion(major=1, minor=0, patch=0),
                capability_type=CapabilityType.STRUCTURED,
                agent_type=AgentType.SOFTWARE,
                inputs=[],
                outputs=[],
                # Missing structured_spec
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_unstructured_capability_requires_spec(self):
        """Test that unstructured capabilities require unstructured_spec."""
        with pytest.raises(ValueError, match="Unstructured capabilities must have unstructured_spec"):
            Capability(
                id="cap-invalid",
                name="invalid",
                description="Invalid unstructured capability",
                version=CapabilityVersion(major=1, minor=0, patch=0),
                capability_type=CapabilityType.UNSTRUCTURED,
                agent_type=AgentType.LLM,
                inputs=[],
                outputs=[],
                # Missing unstructured_spec
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_hybrid_capability_requires_both_specs(self):
        """Test that hybrid capabilities require both specs."""
        with pytest.raises(ValueError, match="Hybrid capabilities must have both"):
            Capability(
                id="cap-invalid",
                name="invalid",
                description="Invalid hybrid capability",
                version=CapabilityVersion(major=1, minor=0, patch=0),
                capability_type=CapabilityType.HYBRID,
                agent_type=AgentType.LLM,
                inputs=[],
                outputs=[],
                structured_spec=StructuredCapability(
                    endpoint="/test",
                    input_schema={},
                    output_schema={},
                ),
                # Missing unstructured_spec
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_is_deprecated(self, sample_structured_capability):
        """Test deprecation check."""
        cap = sample_structured_capability

        assert cap.is_deprecated() is False

        cap.deprecated = True
        assert cap.is_deprecated() is True

    def test_to_summary(self, sample_structured_capability):
        """Test creating capability summary."""
        cap = sample_structured_capability
        summary = cap.to_summary()

        assert summary["id"] == cap.id
        assert summary["name"] == cap.name
        assert summary["version"] == "1.0.0"
        assert summary["type"] == CapabilityType.STRUCTURED
        assert "description" in summary

    def test_to_summary_truncates_long_description(self):
        """Test that summary truncates long descriptions."""
        long_desc = "A" * 150
        cap = Capability(
            id="cap-long",
            name="test",
            description=long_desc,
            version=CapabilityVersion(major=1, minor=0, patch=0),
            capability_type=CapabilityType.STRUCTURED,
            agent_type=AgentType.SOFTWARE,
            inputs=[],
            outputs=[],
            structured_spec=StructuredCapability(
                endpoint="/test",
                input_schema={},
                output_schema={},
            ),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        summary = cap.to_summary()
        assert len(summary["description"]) <= 104  # 100 + "..."

    def test_semantic_metadata(self, sample_structured_capability):
        """Test semantic metadata."""
        cap = sample_structured_capability
        cap.semantic.tags = ["translation", "language", "english", "french"]
        cap.semantic.categories = ["nlp", "translation"]

        assert len(cap.semantic.tags) == 4
        assert "translation" in cap.semantic.tags
        assert "nlp" in cap.semantic.categories

    def test_constraints(self, sample_structured_capability):
        """Test capability constraints."""
        cap = sample_structured_capability
        cap.constraints.max_concurrent = 10
        cap.constraints.rate_limit = {"requests_per_minute": 60}
        cap.constraints.timeout_seconds = 30

        assert cap.constraints.max_concurrent == 10
        assert cap.constraints.rate_limit["requests_per_minute"] == 60
        assert cap.constraints.timeout_seconds == 30
