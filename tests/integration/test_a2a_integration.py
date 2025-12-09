"""Integration tests for A2A adapter."""

import pytest
from datetime import datetime

from capabilitymesh.integrations.a2a import A2AAdapter, A2ADiscoveryBridge, AgentCardConverter
from capabilitymesh.core.types import AgentType, CapabilityType
from capabilitymesh import Capability, CapabilityVersion


class TestA2AAdapter:
    """Tests for A2A adapter."""

    def test_wrap_a2a_agent(self):
        """Test wrapping an A2A agent."""
        adapter = A2AAdapter.wrap(
            agent_url="http://example.com/agent",
            name="TestAgent"
        )

        assert adapter is not None
        assert adapter.agent_identity.name == "TestAgent"
        assert adapter.agent_url == "http://example.com/agent"
        assert adapter.supports_a2a is True

    def test_wrap_with_agent_card(self):
        """Test wrapping with an agent card."""
        agent_card = {
            "name": "TranslationAgent",
            "description": "Translates text",
            "version": "1.0.0",
            "capabilities": ["translation"],
            "url": "http://example.com/translator",
            "supportedProtocols": ["json-rpc"]
        }

        adapter = A2AAdapter.wrap(
            agent_url="http://example.com/translator",
            agent_card=agent_card
        )

        assert adapter.agent_identity.name == "TranslationAgent"
        assert adapter.agent_card == agent_card

    def test_to_agent_card(self):
        """Test converting to A2A agent card."""
        adapter = A2AAdapter.wrap(
            agent_url="http://example.com/agent",
            name="TestAgent"
        )

        agent_card = adapter.to_agent_card()

        assert agent_card is not None
        assert "name" in agent_card
        assert "url" in agent_card
        assert agent_card["url"] == "http://example.com/agent"

    def test_execute_task(self):
        """Test task execution."""
        adapter = A2AAdapter.wrap(
            agent_url="http://example.com/agent",
            name="TestAgent"
        )

        task = {"instructions": "Process data"}
        result = adapter.execute_task(task)

        assert result is not None
        assert "status" in result


class TestAgentCardConverter:
    """Tests for Agent Card converter."""

    @pytest.fixture
    def sample_capability(self):
        """Create a sample capability."""
        from capabilitymesh import (
            Capability, CapabilityVersion, CapabilityType, AgentType,
            CapabilityInputOutput, IOFormat, UnstructuredCapability
        )

        return Capability(
            id="cap-test",
            name="test-capability",
            description="Test capability",
            version=CapabilityVersion(major=1, minor=0, patch=0),
            capability_type=CapabilityType.UNSTRUCTURED,
            agent_type=AgentType.LLM,
            inputs=[
                CapabilityInputOutput(
                    format=IOFormat.TEXT,
                    description="Input text"
                )
            ],
            outputs=[
                CapabilityInputOutput(
                    format=IOFormat.TEXT,
                    description="Output text"
                )
            ],
            unstructured_spec=UnstructuredCapability(
                prompt_template="Process: {text}"
            ),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    @pytest.fixture
    def sample_agent_identity(self):
        """Create a sample agent identity."""
        from capabilitymesh import AgentIdentity, AgentAddress, AgentType
        from uuid import uuid4

        address = AgentAddress(protocol="http", host="localhost", port=8000)
        public_key = f"-----BEGIN PUBLIC KEY-----\n{uuid4().hex}\n-----END PUBLIC KEY-----"
        did = AgentIdentity.generate_did(public_key)

        return AgentIdentity(
            did=did,
            name="TestAgent",
            agent_type=AgentType.LLM,
            addresses=[address],
            primary_address=address,
            public_key=public_key,
            created_at=datetime.now(),
            last_seen=datetime.now()
        )

    def test_capability_to_agent_card(self, sample_capability, sample_agent_identity):
        """Test converting capability to agent card."""
        agent_card = AgentCardConverter.capability_to_agent_card(
            sample_capability,
            sample_agent_identity
        )

        assert agent_card is not None
        assert agent_card["name"] == "TestAgent"
        assert agent_card["version"] == "1.0.0"
        assert "capabilities" in agent_card
        assert "url" in agent_card

    def test_agent_card_to_capability(self):
        """Test converting agent card to capability."""
        agent_card = {
            "name": "TranslationAgent",
            "description": "Translates text",
            "version": "1.0.0",
            "capabilities": ["translation", "language"],
            "url": "http://example.com/translator"
        }

        capability = AgentCardConverter.agent_card_to_capability(agent_card)

        assert capability is not None
        assert capability.name == "TranslationAgent"
        assert capability.version.semver == "1.0.0"
        assert "translation" in capability.semantic.tags


class TestA2ADiscoveryBridge:
    """Tests for A2A discovery bridge."""

    def test_create_discovery_bridge(self):
        """Test creating a discovery bridge."""
        bridge = A2ADiscoveryBridge()
        assert bridge is not None
        assert len(bridge.discovered_agents) == 0

    def test_discover_a2a_agent(self):
        """Test discovering an A2A agent."""
        bridge = A2ADiscoveryBridge()
        adapter = bridge.discover_a2a_agent(
            agent_url="http://example.com/agent",
            fetch_agent_card=False
        )

        assert adapter is not None
        assert len(bridge.discovered_agents) == 1
        assert bridge.discovered_agents[0] == adapter

    def test_get_all_discovered_agents(self):
        """Test getting all discovered agents."""
        bridge = A2ADiscoveryBridge()
        bridge.discover_a2a_agent("http://example.com/agent1", fetch_agent_card=False)
        bridge.discover_a2a_agent("http://example.com/agent2", fetch_agent_card=False)

        agents = bridge.get_all_discovered_agents()
        assert len(agents) == 2
