"""Capability schema definitions for CapabilityMesh."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from capabilitymesh.core.types import AgentType, CapabilityType, IOFormat


class CapabilityVersion(BaseModel):
    """Semantic versioning for capabilities."""

    major: int
    minor: int
    patch: int

    @property
    def semver(self) -> str:
        """Get semantic version string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible(self, other: "CapabilityVersion") -> bool:
        """Check if this version is compatible with another.

        Compatibility follows semantic versioning: same major version = compatible

        Args:
            other: Another CapabilityVersion to compare

        Returns:
            True if compatible (same major version), False otherwise
        """
        return self.major == other.major

    @classmethod
    def from_string(cls, version_str: str) -> "CapabilityVersion":
        """Create CapabilityVersion from semver string.

        Args:
            version_str: Version string like "1.2.3"

        Returns:
            CapabilityVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        try:
            parts = version_str.split(".")
            if len(parts) != 3:
                raise ValueError("Version must have 3 parts (major.minor.patch)")

            major, minor, patch = map(int, parts)
            return cls(major=major, minor=minor, patch=patch)

        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid version string: {version_str}") from e

    def __str__(self) -> str:
        """String representation."""
        return self.semver

    def __lt__(self, other: "CapabilityVersion") -> bool:
        """Less than comparison."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "CapabilityVersion") -> bool:
        """Less than or equal comparison."""
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def __gt__(self, other: "CapabilityVersion") -> bool:
        """Greater than comparison."""
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other: "CapabilityVersion") -> bool:
        """Greater than or equal comparison."""
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)


class CapabilityInputOutput(BaseModel):
    """Defines input/output specification for a capability."""

    format: IOFormat
    json_schema: Optional[Dict[str, Any]] = None  # JSON Schema for structured formats
    description: str
    examples: List[Any] = Field(default_factory=list)

    def validate_data(self, data: Any) -> bool:
        """Validate data against this IO specification.

        Args:
            data: Data to validate

        Returns:
            True if data matches specification, False otherwise
        """
        # Basic format checking
        if self.format == IOFormat.JSON:
            if not isinstance(data, dict):
                return False
            # TODO: Add JSON schema validation if schema is provided
        elif self.format == IOFormat.TEXT:
            if not isinstance(data, str):
                return False

        return True


class StructuredCapability(BaseModel):
    """Structured API-based capability specification."""

    endpoint: str
    method: str = "POST"
    input_schema: Dict[str, Any]  # JSON Schema
    output_schema: Dict[str, Any]  # JSON Schema
    openapi_spec: Optional[str] = None  # OpenAPI/Swagger URL


class UnstructuredCapability(BaseModel):
    """LLM/Natural language capability specification."""

    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    context_window: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class CapabilityConstraints(BaseModel):
    """Resource and operational constraints for a capability."""

    max_concurrent: Optional[int] = None
    rate_limit: Optional[Dict[str, int]] = None  # {"requests_per_minute": 60}
    timeout_seconds: Optional[int] = None
    cost: Optional[Dict[str, Any]] = None  # {"currency": "USD", "per_request": 0.01}
    geographic_restrictions: List[str] = Field(default_factory=list)


class SemanticMetadata(BaseModel):
    """Semantic annotations for capability matching."""

    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    ontology_uris: List[str] = Field(default_factory=list)  # Links to ontologies
    embeddings: Optional[Dict[str, List[float]]] = None  # Pre-computed embeddings
    related_capabilities: List[str] = Field(default_factory=list)  # Capability IDs


class Capability(BaseModel):
    """Core Capability Schema.

    Represents a single capability that an agent can provide or request.
    """

    # Identity
    id: str  # Unique capability identifier (UUID)
    name: str
    description: str  # Natural language description
    version: CapabilityVersion

    # Type classification
    capability_type: CapabilityType
    agent_type: AgentType

    # Input/Output specification
    inputs: List[CapabilityInputOutput]
    outputs: List[CapabilityInputOutput]

    # Type-specific details
    structured_spec: Optional[StructuredCapability] = None
    unstructured_spec: Optional[UnstructuredCapability] = None

    # Constraints and requirements
    constraints: CapabilityConstraints = Field(default_factory=CapabilityConstraints)
    prerequisites: List[str] = Field(default_factory=list)  # Required capability IDs

    # Semantic metadata
    semantic: SemanticMetadata = Field(default_factory=SemanticMetadata)

    # Trust and validation
    attestation: Optional[Dict[str, Any]] = None  # Cryptographic proof
    validation_endpoint: Optional[str] = None  # URL to validate capability

    # Metadata
    created_at: datetime
    updated_at: datetime
    deprecated: bool = False
    deprecation_notice: Optional[str] = None

    @classmethod
    def create_simple(
        cls,
        name: str,
        description: str,
        capability_type: CapabilityType = CapabilityType.UNSTRUCTURED,
        agent_type: AgentType = AgentType.SOFTWARE,
        tags: Optional[List[str]] = None,
    ) -> "Capability":
        """Create a simple capability with auto-generated fields.

        Args:
            name: Capability name
            description: Capability description
            capability_type: Type of capability
            agent_type: Agent type
            tags: Optional semantic tags

        Returns:
            Capability with auto-generated fields
        """
        import uuid
        from datetime import datetime

        cap_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Create basic I/O spec
        basic_io = CapabilityInputOutput(
            format=IOFormat.TEXT,
            description=f"{name} input/output",
        )

        # Create type-specific spec
        structured_spec = None
        unstructured_spec = None

        if capability_type == CapabilityType.STRUCTURED:
            structured_spec = StructuredCapability(
                endpoint="local",
                method="CALL",
                input_schema={},
                output_schema={},
            )
        elif capability_type == CapabilityType.UNSTRUCTURED:
            unstructured_spec = UnstructuredCapability()
        elif capability_type == CapabilityType.HYBRID:
            structured_spec = StructuredCapability(
                endpoint="local",
                method="CALL",
                input_schema={},
                output_schema={},
            )
            unstructured_spec = UnstructuredCapability()

        return cls(
            id=cap_id,
            name=name,
            description=description,
            version=CapabilityVersion(major=1, minor=0, patch=0),
            capability_type=capability_type,
            agent_type=agent_type,
            inputs=[basic_io],
            outputs=[basic_io],
            structured_spec=structured_spec,
            unstructured_spec=unstructured_spec,
            semantic=SemanticMetadata(tags=tags or []),
            created_at=now,
            updated_at=now,
        )

    def model_post_init(self, __context: object) -> None:
        """Validate after model initialization."""
        # Ensure type-specific spec is provided
        if self.capability_type == CapabilityType.STRUCTURED and not self.structured_spec:
            raise ValueError("Structured capabilities must have structured_spec")

        if self.capability_type == CapabilityType.UNSTRUCTURED and not self.unstructured_spec:
            raise ValueError("Unstructured capabilities must have unstructured_spec")

        if self.capability_type == CapabilityType.HYBRID:
            if not self.structured_spec or not self.unstructured_spec:
                raise ValueError("Hybrid capabilities must have both structured and unstructured specs")

    def is_deprecated(self) -> bool:
        """Check if capability is deprecated."""
        return self.deprecated

    def matches_version_requirement(self, requirement: str) -> bool:
        """Check if this capability version matches a requirement.

        Args:
            requirement: Version requirement string like ">=1.0.0,<2.0.0"

        Returns:
            True if version matches requirement, False otherwise
        """
        # Simple implementation - just check compatibility for now
        # TODO: Implement full version range parsing
        if ">=" in requirement or "<" in requirement:
            # For now, just check major version compatibility
            return True

        # Exact version match
        try:
            required_version = CapabilityVersion.from_string(requirement)
            return self.version == required_version
        except ValueError:
            return False

    def to_summary(self) -> Dict[str, Any]:
        """Get a summary representation of the capability.

        Returns:
            Dictionary with key capability information
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description[:100] + "..." if len(self.description) > 100 else self.description,
            "version": self.version.semver,
            "type": self.capability_type,
            "agent_type": self.agent_type,
            "tags": self.semantic.tags,
            "deprecated": self.deprecated,
        }
