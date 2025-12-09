"""Redis-based storage backend for CapabilityMesh.

Provides distributed storage using Redis with:
- Redis hash storage for agents
- JSON serialization
- Optional TTL support
- Async operations using redis.asyncio
- Connection pooling
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from redis.asyncio import Redis
except ImportError:
    raise ImportError(
        "redis is required for RedisStorage. "
        "Install it with: pip install capabilitymesh[redis]"
    )

from .base import AgentRecord, Storage


class RedisStorage(Storage):
    """Redis-based storage backend.

    Features:
    - Distributed storage using Redis
    - JSON serialization for complex data
    - Optional TTL for automatic expiration
    - Async operations with connection pooling
    - Suitable for multi-instance deployments

    Example:
        storage = RedisStorage(host="localhost", port=6379)
        mesh = Mesh(storage=storage)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "capabilitymesh:",
        ttl: Optional[int] = None,
    ):
        """Initialize Redis storage.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Optional Redis password
            prefix: Key prefix for all Redis keys
            ttl: Optional TTL in seconds for automatic expiration
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.ttl = ttl
        self._redis: Optional[Redis] = None

    async def _get_redis(self) -> Redis:
        """Get Redis connection, creating if needed."""
        if self._redis is None:
            self._redis = Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
        return self._redis

    def _agent_key(self, agent_id: str) -> str:
        """Get Redis key for agent data."""
        return f"{self.prefix}agents:{agent_id}"

    def _capability_key(self, cap_id: str) -> str:
        """Get Redis key for capability data."""
        return f"{self.prefix}capabilities:{cap_id}"

    def _agent_ids_key(self) -> str:
        """Get Redis key for agent IDs set."""
        return f"{self.prefix}agent_ids"

    def _trust_key(self, agent_id: str) -> str:
        """Get Redis key for trust scores."""
        return f"{self.prefix}trust:{agent_id}"

    async def save_agent(self, record: AgentRecord) -> None:
        """Save an agent record to Redis.

        Args:
            record: Agent record to save
        """
        redis = await self._get_redis()
        agent_id = record.identity.id

        # Prepare agent data
        agent_data = {
            "id": agent_id,
            "name": record.identity.name,
            "agent_type": record.identity.agent_type.value,
            "description": getattr(record.identity, "description", "") or "",
            "registered_at": record.registered_at.isoformat() if record.registered_at else None,
            "metadata": json.dumps(record.metadata) if record.metadata else "{}",
            "capability_ids": json.dumps([cap.id for cap in record.capabilities]),
        }

        # Save agent data
        agent_key = self._agent_key(agent_id)
        await redis.hset(agent_key, mapping=agent_data)
        if self.ttl:
            await redis.expire(agent_key, self.ttl)

        # Save capabilities
        for cap in record.capabilities:
            # Extract tags and categories from semantic field
            tags = []
            categories = []
            if cap.semantic:
                tags = cap.semantic.tags or []
                categories = cap.semantic.categories or []

            cap_data = {
                "id": cap.id,
                "name": cap.name,
                "description": cap.description or "",
                "type": cap.capability_type.value if cap.capability_type else "unstructured",
                "tags": json.dumps(tags),
                "categories": json.dumps(categories),
            }

            cap_key = self._capability_key(cap.id)
            await redis.hset(cap_key, mapping=cap_data)
            if self.ttl:
                await redis.expire(cap_key, self.ttl)

        # Add to agent IDs set
        await redis.sadd(self._agent_ids_key(), agent_id)

    async def get_agent(self, agent_id: str) -> Optional[AgentRecord]:
        """Retrieve an agent by ID.

        Args:
            agent_id: Agent ID to retrieve

        Returns:
            AgentRecord if found, None otherwise
        """
        redis = await self._get_redis()

        # Get agent data
        agent_key = self._agent_key(agent_id)
        agent_data = await redis.hgetall(agent_key)

        if not agent_data:
            return None

        # Get capability IDs
        cap_ids = json.loads(agent_data.get("capability_ids", "[]"))

        # Get capabilities
        capabilities = []
        for cap_id in cap_ids:
            cap_key = self._capability_key(cap_id)
            cap_data = await redis.hgetall(cap_key)
            if cap_data:
                from capabilitymesh.schemas.capability import Capability, CapabilityType

                cap = Capability.create_simple(
                    name=cap_data["name"],
                    description=cap_data.get("description", ""),
                    capability_type=CapabilityType(cap_data.get("type", "unstructured")),
                    tags=json.loads(cap_data.get("tags", "[]")),
                )
                cap.id = cap_data["id"]

                # Update semantic categories if present
                categories = json.loads(cap_data.get("categories", "[]"))
                if categories and cap.semantic:
                    cap.semantic.categories = categories

                capabilities.append(cap)

        # Build identity
        from capabilitymesh.core.identity import AgentIdentity
        from capabilitymesh.core.types import AgentType

        identity = AgentIdentity.create_simple(
            name=agent_data["name"],
            agent_type=AgentType(agent_data["agent_type"]),
            description=agent_data.get("description", ""),
        )
        identity.did = agent_data["id"]

        # Parse metadata and registered_at
        metadata = json.loads(agent_data.get("metadata", "{}"))
        registered_at = None
        if agent_data.get("registered_at"):
            try:
                registered_at = datetime.fromisoformat(agent_data["registered_at"])
            except (ValueError, TypeError):
                pass

        return AgentRecord(
            identity=identity,
            capabilities=capabilities,
            metadata=metadata,
            registered_at=registered_at,
        )

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID.

        Args:
            agent_id: Agent ID to delete

        Returns:
            True if deleted, False if not found
        """
        redis = await self._get_redis()

        # Check if agent exists
        agent_key = self._agent_key(agent_id)
        exists = await redis.exists(agent_key)

        if not exists:
            return False

        # Get capability IDs
        agent_data = await redis.hgetall(agent_key)
        cap_ids = json.loads(agent_data.get("capability_ids", "[]"))

        # Delete capabilities
        for cap_id in cap_ids:
            cap_key = self._capability_key(cap_id)
            await redis.delete(cap_key)

        # Delete agent data
        await redis.delete(agent_key)

        # Remove from agent IDs set
        await redis.srem(self._agent_ids_key(), agent_id)

        # Delete trust data if exists
        trust_key = self._trust_key(agent_id)
        await redis.delete(trust_key)

        return True

    async def list_all(self) -> List[AgentRecord]:
        """List all agents in storage.

        Returns:
            List of all agent records
        """
        redis = await self._get_redis()

        # Get all agent IDs
        agent_ids = await redis.smembers(self._agent_ids_key())

        # Get all agents
        results = []
        for agent_id in agent_ids:
            agent = await self.get_agent(agent_id)
            if agent:
                results.append(agent)

        return results

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[AgentRecord]:
        """Search for agents matching the query.

        Note: Redis doesn't have built-in FTS, so this does in-memory filtering.
        For production use with large datasets, consider RediSearch module.

        Args:
            query: Search query string
            limit: Maximum number of results
            filters: Optional filters (e.g., agent_type, capabilities)

        Returns:
            List of matching agent records
        """
        # Get all agents
        all_agents = await self.list_all()

        # Filter by query (simple substring match)
        query_lower = query.lower()
        matching_agents = []

        for agent in all_agents:
            # Search in agent name
            if query_lower in agent.identity.name.lower():
                matching_agents.append(agent)
                continue

            # Search in agent description
            desc = getattr(agent.identity, "description", "") or ""
            if query_lower in desc.lower():
                matching_agents.append(agent)
                continue

            # Search in capability names and descriptions
            for cap in agent.capabilities:
                if query_lower in cap.name.lower():
                    matching_agents.append(agent)
                    break
                if query_lower in (cap.description or "").lower():
                    matching_agents.append(agent)
                    break

        # Apply filters
        if filters:
            filtered_agents = []
            for agent in matching_agents:
                if self._matches_filters(agent, filters):
                    filtered_agents.append(agent)
            matching_agents = filtered_agents

        # Return limited results
        return matching_agents[:limit]

    async def update_metadata(
        self, agent_id: str, metadata: Dict[str, Any]
    ) -> bool:
        """Update an agent's metadata.

        Args:
            agent_id: Agent ID to update
            metadata: Metadata dictionary to merge

        Returns:
            True if updated, False if not found
        """
        redis = await self._get_redis()

        # Get current agent data
        agent_key = self._agent_key(agent_id)
        agent_data = await redis.hgetall(agent_key)

        if not agent_data:
            return False

        # Merge metadata
        current_metadata = json.loads(agent_data.get("metadata", "{}"))
        current_metadata.update(metadata)

        # Update metadata
        await redis.hset(agent_key, "metadata", json.dumps(current_metadata))

        return True

    def _matches_filters(
        self, record: AgentRecord, filters: Dict[str, Any]
    ) -> bool:
        """Check if a record matches the given filters.

        Args:
            record: Agent record to check
            filters: Filters to apply

        Returns:
            True if record matches all filters
        """
        # Filter by capabilities
        if "capabilities" in filters:
            required_caps = set(filters["capabilities"])
            agent_caps = {cap.name for cap in record.capabilities}
            if not required_caps.issubset(agent_caps):
                return False

        # Filter by tags
        if "tags" in filters:
            required_tags = set(filters["tags"])
            agent_tags = set()
            for cap in record.capabilities:
                if cap.semantic:
                    agent_tags.update(cap.semantic.tags or [])
            if not required_tags.issubset(agent_tags):
                return False

        # Filter by agent_type
        if "agent_type" in filters:
            if record.identity.agent_type.value != filters["agent_type"]:
                return False

        return True

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
