"""Converter between ACDP Capabilities and A2A Agent Cards.

The A2A Agent Card format is a JSON structure that describes an agent's
capabilities. This module provides bidirectional conversion between ACDP's
rich capability schema and A2A's Agent Card format.

A2A Agent Card Example:
{
    "name": "TranslationAgent",
    "description": "Translates text between languages",
    "version": "1.0.0",
    "capabilities": ["translation", "language-detection"],
    "url": "https://example.com/agent",
    "supportedProtocols": ["json-rpc"],
    "metadata": {...}
}
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from capabilitymesh.core.identity import AgentIdentity, AgentAddress
from capabilitymesh.core.types import AgentType, CapabilityType, IOFormat
from capabilitymesh.schemas.capability import (
    Capability,
    CapabilityVersion,
    CapabilityInputOutput,
    StructuredCapability,
    UnstructuredCapability,
)


class AgentCardConverter:
    """Converter between ACDP and A2A formats."""

    @staticmethod
    def capability_to_agent_card(
        capability: Capability,
        agent_identity: AgentIdentity
    ) -> Dict[str, Any]:
        """Convert ACDP Capability to A2A Agent Card format.

        Args:
            capability: ACDP Capability to convert
            agent_identity: Agent providing this capability

        Returns:
            A2A Agent Card as dictionary
        """
        # Extract capabilities list from semantic metadata
        capabilities_list = list(set(
            capability.semantic.tags +
            capability.semantic.categories +
            [capability.name]
        ))

        # Build metadata from ACDP capability
        metadata = {
            "acdp": {
                "capability_id": capability.id,
                "capability_type": capability.capability_type,
                "agent_type": capability.agent_type,
                "version": capability.version.semver,
                "deprecated": capability.deprecated,
            }
        }

        # Add cost information if available
        if capability.constraints.cost:
            metadata["cost"] = capability.constraints.cost

        # Add rate limit if available
        if capability.constraints.rate_limit:
            metadata["rateLimit"] = capability.constraints.rate_limit

        # Build agent card
        agent_card = {
            "name": agent_identity.name,
            "description": capability.description,
            "version": capability.version.semver,
            "capabilities": capabilities_list,
            "url": agent_identity.primary_address.to_uri(),
            "supportedProtocols": ["json-rpc"],  # A2A default
            "metadata": metadata,
        }

        # Add structured API details if available
        if capability.structured_spec:
            agent_card["api"] = {
                "endpoint": capability.structured_spec.endpoint,
                "method": capability.structured_spec.method,
                "inputSchema": capability.structured_spec.input_schema,
                "outputSchema": capability.structured_spec.output_schema,
            }

        # Add LLM details if available
        if capability.unstructured_spec:
            agent_card["llm"] = {
                "modelInfo": capability.unstructured_spec.model_info,
                "contextWindow": capability.unstructured_spec.context_window,
            }

        return agent_card

    @staticmethod
    def agent_card_to_capability(
        agent_card: Dict[str, Any],
        capability_id: Optional[str] = None
    ) -> Capability:
        """Convert A2A Agent Card to ACDP Capability.

        Args:
            agent_card: A2A Agent Card dictionary
            capability_id: Optional capability ID (generated if not provided)

        Returns:
            ACDP Capability instance
        """
        # Generate ID if not provided
        if not capability_id:
            capability_id = f"cap-a2a-{uuid4().hex[:8]}"

        # Parse version
        version_str = agent_card.get("version", "1.0.0")
        version = CapabilityVersion.from_string(version_str)

        # Determine capability type from metadata or API structure
        metadata = agent_card.get("metadata", {})
        acdp_metadata = metadata.get("acdp", {})

        capability_type_str = acdp_metadata.get("capability_type")
        if capability_type_str:
            capability_type = CapabilityType(capability_type_str)
        elif "api" in agent_card:
            capability_type = CapabilityType.STRUCTURED
        elif "llm" in agent_card:
            capability_type = CapabilityType.UNSTRUCTURED
        else:
            # If no spec is available, default to UNSTRUCTURED (more flexible)
            capability_type = CapabilityType.UNSTRUCTURED

        # Determine agent type
        agent_type_str = acdp_metadata.get("agent_type")
        if agent_type_str:
            agent_type = AgentType(agent_type_str)
        elif "llm" in agent_card:
            agent_type = AgentType.LLM
        else:
            agent_type = AgentType.SOFTWARE

        # Create basic inputs/outputs
        # For A2A cards, we create generic IO specs
        inputs = [
            CapabilityInputOutput(
                format=IOFormat.JSON,
                description="Input data for A2A task",
                json_schema=agent_card.get("api", {}).get("inputSchema"),
            )
        ]

        outputs = [
            CapabilityInputOutput(
                format=IOFormat.JSON,
                description="Output data from A2A task",
                json_schema=agent_card.get("api", {}).get("outputSchema"),
            )
        ]

        # Create structured spec if API is defined
        structured_spec = None
        if "api" in agent_card:
            api = agent_card["api"]
            structured_spec = StructuredCapability(
                endpoint=api.get("endpoint", "/"),
                method=api.get("method", "POST"),
                input_schema=api.get("inputSchema", {}),
                output_schema=api.get("outputSchema", {}),
            )

        # Create unstructured spec if LLM is defined or if type is UNSTRUCTURED
        unstructured_spec = None
        if "llm" in agent_card:
            llm = agent_card["llm"]
            unstructured_spec = UnstructuredCapability(
                model_info=llm.get("modelInfo"),
                context_window=llm.get("contextWindow"),
            )
        elif capability_type == CapabilityType.UNSTRUCTURED:
            # Create a basic unstructured spec for A2A agents
            unstructured_spec = UnstructuredCapability(
                model_info={"source": "a2a", "name": agent_card.get("name")},
            )

        # Create capability
        now = datetime.now()
        capability = Capability(
            id=capability_id,
            name=agent_card.get("name", "unknown"),
            description=agent_card.get("description", ""),
            version=version,
            capability_type=capability_type,
            agent_type=agent_type,
            inputs=inputs,
            outputs=outputs,
            structured_spec=structured_spec,
            unstructured_spec=unstructured_spec,
            created_at=now,
            updated_at=now,
            deprecated=acdp_metadata.get("deprecated", False),
        )

        # Add semantic metadata from capabilities list
        capabilities_list = agent_card.get("capabilities", [])
        capability.semantic.tags = capabilities_list

        return capability

    @staticmethod
    def capabilities_to_agent_card_list(
        capabilities: List[Capability],
        agent_identity: AgentIdentity
    ) -> List[Dict[str, Any]]:
        """Convert multiple ACDP capabilities to A2A Agent Card list.

        Args:
            capabilities: List of ACDP Capabilities
            agent_identity: Agent providing these capabilities

        Returns:
            List of A2A Agent Cards
        """
        return [
            AgentCardConverter.capability_to_agent_card(cap, agent_identity)
            for cap in capabilities
        ]


# Convenience functions for direct conversion
def capability_to_agent_card(
    capability: Capability,
    agent_identity: AgentIdentity
) -> Dict[str, Any]:
    """Convert ACDP Capability to A2A Agent Card.

    Args:
        capability: ACDP Capability
        agent_identity: Agent identity

    Returns:
        A2A Agent Card dictionary
    """
    return AgentCardConverter.capability_to_agent_card(capability, agent_identity)


def agent_card_to_capability(
    agent_card: Dict[str, Any],
    capability_id: Optional[str] = None
) -> Capability:
    """Convert A2A Agent Card to ACDP Capability.

    Args:
        agent_card: A2A Agent Card dictionary
        capability_id: Optional capability ID

    Returns:
        ACDP Capability instance
    """
    return AgentCardConverter.agent_card_to_capability(agent_card, capability_id)
