"""In-memory storage backend for CapabilityMesh."""

from typing import Any, Dict, List, Optional
from collections import defaultdict

from .base import AgentRecord, Storage


class InMemoryStorage(Storage):
    """In-memory storage backend.

    Provides fast, zero-configuration storage for agent records.
    Data is lost when the process terminates.

    Thread-safe for async operations.
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._agents: Dict[str, AgentRecord] = {}
        self._capability_index: Dict[str, set] = defaultdict(set)

    async def save_agent(self, record: AgentRecord) -> None:
        """Save an agent record.

        Args:
            record: The agent record to save
        """
        agent_id = record.identity.id
        self._agents[agent_id] = record

        # Index capabilities for fast lookup
        for capability in record.capabilities:
            cap_name = capability.name.lower()
            self._capability_index[cap_name].add(agent_id)

            # Also index tags and categories
            if hasattr(capability, "semantic") and capability.semantic:
                for tag in capability.semantic.tags:
                    self._capability_index[tag.lower()].add(agent_id)
                for category in capability.semantic.categories:
                    self._capability_index[category.lower()].add(agent_id)

    async def get_agent(self, agent_id: str) -> Optional[AgentRecord]:
        """Retrieve an agent by ID.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            The agent record if found, None otherwise
        """
        return self._agents.get(agent_id)

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            True if deleted, False if not found
        """
        if agent_id not in self._agents:
            return False

        record = self._agents[agent_id]

        # Remove from capability index
        for capability in record.capabilities:
            cap_name = capability.name.lower()
            self._capability_index[cap_name].discard(agent_id)

            if hasattr(capability, "semantic") and capability.semantic:
                for tag in capability.semantic.tags:
                    self._capability_index[tag.lower()].discard(agent_id)
                for category in capability.semantic.categories:
                    self._capability_index[category.lower()].discard(agent_id)

        del self._agents[agent_id]
        return True

    async def list_all(self) -> List[AgentRecord]:
        """List all registered agents.

        Returns:
            List of all agent records
        """
        return list(self._agents.values())

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[AgentRecord]:
        """Search for agents matching the query.

        Uses keyword-based search on capabilities, tags, and agent names.

        Args:
            query: Search query string
            limit: Maximum number of results
            filters: Optional filters (e.g., agent_type, capabilities)

        Returns:
            List of matching agent records
        """
        query_lower = query.lower()
        matching_ids = set()

        # Search by capability name/tag
        for keyword in query_lower.split():
            if keyword in self._capability_index:
                matching_ids.update(self._capability_index[keyword])

        # Search by agent name and description
        for agent_id, record in self._agents.items():
            if query_lower in record.identity.name.lower():
                matching_ids.add(agent_id)

            # Search in capability descriptions
            for cap in record.capabilities:
                if query_lower in cap.description.lower():
                    matching_ids.add(agent_id)
                    break

        # Apply filters
        results = []
        for agent_id in matching_ids:
            record = self._agents[agent_id]

            if filters:
                # Filter by agent_type
                if "agent_type" in filters:
                    if record.identity.agent_type.value != filters["agent_type"]:
                        continue

                # Filter by capabilities
                if "capabilities" in filters:
                    required_caps = set(filters["capabilities"])
                    agent_caps = {cap.name.lower() for cap in record.capabilities}
                    if not required_caps.issubset(agent_caps):
                        continue

            results.append(record)

            if len(results) >= limit:
                break

        return results

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
        if agent_id not in self._agents:
            return False

        self._agents[agent_id].metadata.update(metadata)
        return True

    def clear(self) -> None:
        """Clear all stored data. Useful for testing."""
        self._agents.clear()
        self._capability_index.clear()
