"""Adapter for wrapping A2A agents with ACDP capabilities."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from capabilitymesh.core.identity import AgentIdentity, AgentAddress
from capabilitymesh.core.types import AgentType
from capabilitymesh.schemas.capability import Capability
from capabilitymesh.integrations.base import A2ACompatibleAdapter
from capabilitymesh.integrations.a2a.converter import AgentCardConverter
from capabilitymesh.integrations.a2a.client import A2AClient


class A2AAdapter(A2ACompatibleAdapter):
    """Adapter for A2A protocol agents.

    This adapter wraps A2A-compatible agents, making them discoverable
    via ACDP while maintaining A2A protocol compatibility for execution.
    """

    def __init__(
        self,
        agent_identity: AgentIdentity,
        agent_url: str,
        agent_card: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize A2A adapter.

        Args:
            agent_identity: ACDP agent identity
            agent_url: URL of the A2A agent
            agent_card: Optional A2A Agent Card (fetched if not provided)
        """
        # Create a placeholder framework agent (A2A client)
        framework_agent = A2AClient(agent_url=agent_url)

        super().__init__(agent_identity, framework_agent)

        self.agent_url = agent_url
        self.agent_card = agent_card
        self.a2a_client = framework_agent

    @classmethod
    def wrap(
        cls,
        agent_url: str,
        name: Optional[str] = None,
        agent_card: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> "A2AAdapter":
        """Wrap an A2A agent for ACDP discovery.

        Args:
            agent_url: URL of the A2A agent
            name: Optional name for the agent
            agent_card: Optional A2A Agent Card
            **kwargs: Additional arguments

        Returns:
            A2AAdapter instance
        """
        # Parse URL to create AgentAddress
        address = AgentAddress.from_uri(agent_url)

        # If agent card is provided, extract name
        if agent_card and not name:
            name = agent_card.get("name", "A2AAgent")

        # Create agent identity
        # In production, this should fetch the agent's public key
        # For now, generate a unique key based on agent_url to ensure unique DIDs
        import hashlib
        unique_key = hashlib.sha256(agent_url.encode()).hexdigest()
        public_key = f"-----BEGIN PUBLIC KEY-----\n{unique_key}\n-----END PUBLIC KEY-----"
        did = AgentIdentity.generate_did(public_key)

        agent_identity = AgentIdentity(
            did=did,
            name=name or "A2AAgent",
            agent_type=AgentType.SOFTWARE,  # Can be inferred from agent card
            addresses=[address],
            primary_address=address,
            public_key=public_key,
            created_at=datetime.now(),
            last_seen=datetime.now(),
        )

        return cls(
            agent_identity=agent_identity,
            agent_url=agent_url,
            agent_card=agent_card
        )

    def extract_capabilities(self) -> List[Capability]:
        """Extract capabilities from A2A Agent Card.

        Returns:
            List of ACDP Capabilities extracted from Agent Card
        """
        if not self.agent_card:
            # In production, fetch agent card from agent_url
            # For now, return empty list
            return []

        # Convert A2A Agent Card to ACDP Capability
        capability = AgentCardConverter.agent_card_to_capability(self.agent_card)
        return [capability]

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using A2A protocol.

        Args:
            task: Task specification

        Returns:
            Task result
        """
        # Create A2A task
        a2a_task = self.a2a_client.create_task(
            instructions=task.get("instructions", ""),
            metadata=task.get("metadata", {})
        )

        # Execute via A2A (placeholder - requires async implementation)
        # In production, this would be: await self.a2a_client.execute_task(a2a_task)
        return {
            "status": "pending",
            "message": "A2A execution requires async implementation"
        }

    def to_agent_card(self) -> Dict[str, Any]:
        """Convert ACDP capabilities to A2A Agent Card.

        Returns:
            A2A Agent Card dictionary
        """
        if self.agent_card:
            return self.agent_card

        # If no agent card, create one from ACDP capabilities
        if not self.capabilities:
            # Default agent card
            return {
                "name": self.agent_identity.name,
                "description": self.agent_identity.description or "",
                "version": "1.0.0",
                "capabilities": [],
                "url": self.agent_url,
                "supportedProtocols": ["json-rpc"],
            }

        # Use first capability to create agent card
        return AgentCardConverter.capability_to_agent_card(
            self.capabilities[0],
            self.agent_identity
        )

    def execute_a2a_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an A2A task.

        Args:
            task: A2A task specification

        Returns:
            A2A task result
        """
        return self.execute_task(task)


class A2ADiscoveryBridge:
    """Bridge for discovering A2A agents via ACDP.

    This class helps discover A2A agents and wrap them with ACDP capabilities.
    """

    def __init__(self) -> None:
        """Initialize discovery bridge."""
        self.discovered_agents: List[A2AAdapter] = []

    def discover_a2a_agent(
        self,
        agent_url: str,
        fetch_agent_card: bool = True
    ) -> A2AAdapter:
        """Discover an A2A agent and wrap it for ACDP.

        Args:
            agent_url: URL of the A2A agent
            fetch_agent_card: Whether to fetch the agent card

        Returns:
            A2AAdapter wrapping the discovered agent
        """
        # In production, fetch agent card from agent_url
        agent_card = None
        if fetch_agent_card:
            # Placeholder - would fetch via HTTP
            pass

        adapter = A2AAdapter.wrap(agent_url=agent_url, agent_card=agent_card)

        # Extract capabilities
        adapter.register_auto_capabilities()

        self.discovered_agents.append(adapter)
        return adapter

    def get_all_discovered_agents(self) -> List[A2AAdapter]:
        """Get all discovered A2A agents.

        Returns:
            List of A2AAdapter instances
        """
        return self.discovered_agents
