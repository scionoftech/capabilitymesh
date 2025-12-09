"""Unit tests for RedisStorage backend.

Note: These tests require a running Redis instance.
Tests will be skipped if Redis is not available.
"""

import pytest
from datetime import datetime

from capabilitymesh.core.identity import AgentIdentity
from capabilitymesh.core.types import AgentType
from capabilitymesh.schemas.capability import Capability
from capabilitymesh.storage.base import AgentRecord

try:
    from capabilitymesh.storage.redis import RedisStorage
    from redis.asyncio import Redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# Check if Redis is running
@pytest.fixture(scope="session")
async def redis_available():
    """Check if Redis is available."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not installed")

    try:
        redis = Redis(host="localhost", port=6379, decode_responses=True)
        await redis.ping()
        await redis.close()
        return True
    except Exception:
        pytest.skip("Redis server not running")


@pytest.fixture
async def redis_storage(redis_available):
    """Create a RedisStorage instance for testing."""
    storage = RedisStorage(
        host="localhost",
        port=6379,
        db=15,  # Use a separate test database
        prefix="test:capabilitymesh:",
    )
    yield storage

    # Cleanup: delete all test keys
    redis = await storage._get_redis()
    keys = await redis.keys(f"{storage.prefix}*")
    if keys:
        await redis.delete(*keys)
    await storage.close()


@pytest.fixture
def sample_agent_record():
    """Create a sample agent record for testing."""
    identity = AgentIdentity.create_simple(
        name="test-agent",
        agent_type=AgentType.SOFTWARE,
        description="A test agent",
    )

    capabilities = [
        Capability.create_simple(
            name="text-summarization",
            description="Summarize text",
            tags=["nlp", "text"],
        ),
        Capability.create_simple(
            name="translation",
            description="Translate text",
            tags=["nlp", "language"],
        ),
    ]

    return AgentRecord(
        identity=identity,
        capabilities=capabilities,
        registered_at=datetime.now(),
        metadata={"version": "1.0", "status": "active"},
    )


@pytest.mark.asyncio
async def test_save_and_retrieve_agent(redis_storage, sample_agent_record):
    """Test saving and retrieving an agent."""
    # Save agent
    await redis_storage.save_agent(sample_agent_record)

    # Retrieve agent
    retrieved = await redis_storage.get_agent(sample_agent_record.identity.id)

    assert retrieved is not None
    assert retrieved.identity.id == sample_agent_record.identity.id
    assert retrieved.identity.name == sample_agent_record.identity.name
    assert len(retrieved.capabilities) == len(sample_agent_record.capabilities)
    assert retrieved.metadata["version"] == "1.0"


@pytest.mark.asyncio
async def test_get_nonexistent_agent(redis_storage):
    """Test retrieving a non-existent agent returns None."""
    result = await redis_storage.get_agent("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_delete_agent(redis_storage, sample_agent_record):
    """Test deleting an agent."""
    # Save agent
    await redis_storage.save_agent(sample_agent_record)

    # Delete agent
    result = await redis_storage.delete_agent(sample_agent_record.identity.id)
    assert result is True

    # Verify deleted
    retrieved = await redis_storage.get_agent(sample_agent_record.identity.id)
    assert retrieved is None


@pytest.mark.asyncio
async def test_list_all_agents(redis_storage):
    """Test listing all agents."""
    # Initially empty
    agents = await redis_storage.list_all()
    assert len(agents) == 0

    # Add multiple agents
    for i in range(3):
        identity = AgentIdentity.create_simple(
            name=f"agent-{i}",
            agent_type=AgentType.SOFTWARE,
        )
        record = AgentRecord(
            identity=identity,
            capabilities=[],
            registered_at=datetime.now(),
        )
        await redis_storage.save_agent(record)

    # List all
    agents = await redis_storage.list_all()
    assert len(agents) == 3
    assert {agent.identity.name for agent in agents} == {"agent-0", "agent-1", "agent-2"}


@pytest.mark.asyncio
async def test_search_agents(redis_storage):
    """Test search functionality."""
    # Create agents
    identity1 = AgentIdentity.create_simple(
        name="summarizer",
        agent_type=AgentType.SOFTWARE,
        description="Summarizes text",
    )
    record1 = AgentRecord(
        identity=identity1,
        capabilities=[
            Capability.create_simple(name="summarization", description="Text summarization")
        ],
        registered_at=datetime.now(),
    )

    identity2 = AgentIdentity.create_simple(
        name="translator",
        agent_type=AgentType.SOFTWARE,
        description="Translates text",
    )
    record2 = AgentRecord(
        identity=identity2,
        capabilities=[
            Capability.create_simple(name="translation", description="Language translation")
        ],
        registered_at=datetime.now(),
    )

    await redis_storage.save_agent(record1)
    await redis_storage.save_agent(record2)

    # Search by name
    results = await redis_storage.search("summarizer", limit=10)
    assert len(results) >= 1
    assert any(agent.identity.name == "summarizer" for agent in results)

    # Search by capability
    results = await redis_storage.search("translation", limit=10)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_update_metadata(redis_storage, sample_agent_record):
    """Test updating agent metadata."""
    # Save agent
    await redis_storage.save_agent(sample_agent_record)

    # Update metadata
    result = await redis_storage.update_metadata(
        sample_agent_record.identity.id, {"version": "2.0", "new_field": "value"}
    )
    assert result is True

    # Verify update
    retrieved = await redis_storage.get_agent(sample_agent_record.identity.id)
    assert retrieved.metadata["version"] == "2.0"
    assert retrieved.metadata["new_field"] == "value"


@pytest.mark.asyncio
async def test_search_with_filters(redis_storage):
    """Test search with capability filters."""
    # Create agents
    identity1 = AgentIdentity.create_simple(name="agent1", agent_type=AgentType.SOFTWARE)
    record1 = AgentRecord(
        identity=identity1,
        capabilities=[Capability.create_simple(name="nlp", description="NLP")],
        registered_at=datetime.now(),
    )

    identity2 = AgentIdentity.create_simple(name="agent2", agent_type=AgentType.SOFTWARE)
    record2 = AgentRecord(
        identity=identity2,
        capabilities=[Capability.create_simple(name="vision", description="Vision")],
        registered_at=datetime.now(),
    )

    await redis_storage.save_agent(record1)
    await redis_storage.save_agent(record2)

    # Search with filter
    results = await redis_storage.search(
        "agent", limit=10, filters={"capabilities": ["nlp"]}
    )
    assert len(results) >= 1
    assert all("nlp" in [c.name for c in agent.capabilities] for agent in results)
