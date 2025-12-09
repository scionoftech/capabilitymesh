"""Unit tests for storage backends."""

import pytest
from datetime import datetime

from capabilitymesh.storage import InMemoryStorage, AgentRecord
from capabilitymesh.core.identity import AgentIdentity, AgentAddress
from capabilitymesh.core.types import AgentType
from capabilitymesh.schemas.capability import Capability
from capabilitymesh.core.types import CapabilityType


@pytest.fixture
def storage():
    """Create a fresh in-memory storage instance."""
    return InMemoryStorage()


@pytest.fixture
def sample_identity():
    """Create a sample agent identity."""
    return AgentIdentity.create_simple(
        name="test-agent",
        agent_type=AgentType.SOFTWARE,
        description="A test agent",
    )


@pytest.fixture
def sample_capability():
    """Create a sample capability."""
    return Capability.create_simple(
        name="test-capability",
        description="A test capability",
        capability_type=CapabilityType.STRUCTURED,
        tags=["test", "sample"],
    )


@pytest.fixture
def sample_record(sample_identity, sample_capability):
    """Create a sample agent record."""
    return AgentRecord(
        identity=sample_identity,
        capabilities=[sample_capability],
        metadata={"key": "value"},
    )


@pytest.mark.asyncio
async def test_save_and_get_agent(storage, sample_record):
    """Test saving and retrieving an agent."""
    # Save agent
    await storage.save_agent(sample_record)

    # Retrieve agent
    retrieved = await storage.get_agent(sample_record.identity.id)

    assert retrieved is not None
    assert retrieved.identity.id == sample_record.identity.id
    assert retrieved.identity.name == "test-agent"
    assert len(retrieved.capabilities) == 1
    assert retrieved.capabilities[0].name == "test-capability"


@pytest.mark.asyncio
async def test_get_nonexistent_agent(storage):
    """Test retrieving a non-existent agent."""
    result = await storage.get_agent("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_list_all_agents(storage, sample_record):
    """Test listing all agents."""
    # Initially empty
    agents = await storage.list_all()
    assert len(agents) == 0

    # Add agents
    await storage.save_agent(sample_record)

    # Create another agent
    identity2 = AgentIdentity.create_simple(
        name="agent-2",
        agent_type=AgentType.LLM,
    )
    record2 = AgentRecord(
        identity=identity2,
        capabilities=[],
        metadata={},
    )
    await storage.save_agent(record2)

    # List all
    agents = await storage.list_all()
    assert len(agents) == 2


@pytest.mark.asyncio
async def test_delete_agent(storage, sample_record):
    """Test deleting an agent."""
    # Save agent
    await storage.save_agent(sample_record)

    # Delete agent
    result = await storage.delete_agent(sample_record.identity.id)
    assert result is True

    # Verify deletion
    retrieved = await storage.get_agent(sample_record.identity.id)
    assert retrieved is None


@pytest.mark.asyncio
async def test_delete_nonexistent_agent(storage):
    """Test deleting a non-existent agent."""
    result = await storage.delete_agent("nonexistent-id")
    assert result is False


@pytest.mark.asyncio
async def test_search_by_capability_name(storage, sample_record):
    """Test searching by capability name."""
    await storage.save_agent(sample_record)

    # Search by exact capability name
    results = await storage.search("test-capability")
    assert len(results) == 1
    assert results[0].identity.id == sample_record.identity.id


@pytest.mark.asyncio
async def test_search_by_tag(storage, sample_record):
    """Test searching by semantic tags."""
    await storage.save_agent(sample_record)

    # Search by tag
    results = await storage.search("test")
    assert len(results) == 1

    results = await storage.search("sample")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_by_agent_name(storage, sample_record):
    """Test searching by agent name."""
    await storage.save_agent(sample_record)

    results = await storage.search("test-agent")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_with_limit(storage):
    """Test search result limit."""
    # Create multiple agents
    for i in range(10):
        identity = AgentIdentity.create_simple(
            name=f"agent-{i}",
            agent_type=AgentType.SOFTWARE,
        )
        capability = Capability.create_simple(
            name="common-capability",
            description="Common capability",
            capability_type=CapabilityType.STRUCTURED,
        )
        record = AgentRecord(identity=identity, capabilities=[capability])
        await storage.save_agent(record)

    # Search with limit
    results = await storage.search("common-capability", limit=5)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_update_metadata(storage, sample_record):
    """Test updating agent metadata."""
    await storage.save_agent(sample_record)

    # Update metadata
    new_metadata = {"new_key": "new_value", "another": 123}
    result = await storage.update_metadata(sample_record.identity.id, new_metadata)
    assert result is True

    # Verify update
    retrieved = await storage.get_agent(sample_record.identity.id)
    assert "new_key" in retrieved.metadata
    assert retrieved.metadata["new_key"] == "new_value"
    assert "key" in retrieved.metadata  # Original key should still exist


@pytest.mark.asyncio
async def test_update_metadata_nonexistent_agent(storage):
    """Test updating metadata for non-existent agent."""
    result = await storage.update_metadata("nonexistent", {"key": "value"})
    assert result is False


def test_clear(storage, sample_record):
    """Test clearing storage."""
    import asyncio

    # Add some data
    asyncio.run(storage.save_agent(sample_record))

    # Clear
    storage.clear()

    # Verify empty
    agents = asyncio.run(storage.list_all())
    assert len(agents) == 0
