"""Storage backend abstraction for CapabilityMesh."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.identity import AgentIdentity
from ..schemas.capability import Capability


class AgentRecord:
    """Record of a registered agent with metadata."""

    def __init__(
        self,
        identity: AgentIdentity,
        capabilities: List[Capability],
        native_agent: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        registered_at: Optional[datetime] = None,
    ):
        self.identity = identity
        self.capabilities = capabilities
        self.native_agent = native_agent
        self.metadata = metadata or {}
        self.registered_at = registered_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "identity": {
                "id": self.identity.id,
                "name": self.identity.name,
                "agent_type": self.identity.agent_type.value,
                "public_key": self.identity.public_key,
                "addresses": [addr.to_dict() for addr in self.identity.addresses],
            },
            "capabilities": [cap.model_dump() for cap in self.capabilities],
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat(),
        }


class Storage(ABC):
    """Abstract base class for agent storage backends."""

    @abstractmethod
    async def save_agent(self, record: AgentRecord) -> None:
        """Save an agent record.

        Args:
            record: The agent record to save
        """
        pass

    @abstractmethod
    async def get_agent(self, agent_id: str) -> Optional[AgentRecord]:
        """Retrieve an agent by ID.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            The agent record if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_all(self) -> List[AgentRecord]:
        """List all registered agents.

        Returns:
            List of all agent records
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[AgentRecord]:
        """Search for agents matching the query.

        Args:
            query: Search query string
            limit: Maximum number of results
            filters: Optional filters (e.g., agent_type, capabilities)

        Returns:
            List of matching agent records
        """
        pass

    @abstractmethod
    async def update_metadata(
        self, agent_id: str, metadata: Dict[str, Any]
    ) -> bool:
        """Update an agent's metadata.

        Args:
            agent_id: The agent's unique identifier
            metadata: Metadata dictionary to merge

        Returns:
            True if updated, False if not found
        """
        pass
