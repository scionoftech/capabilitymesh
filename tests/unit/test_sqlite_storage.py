"""Unit tests for SQLiteStorage backend."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from capabilitymesh.core.identity import AgentIdentity
from capabilitymesh.core.types import AgentType
from capabilitymesh.schemas.capability import Capability
from capabilitymesh.storage.base import AgentRecord

try:
    from capabilitymesh.storage.sqlite import SQLiteStorage

    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not SQLITE_AVAILABLE, reason="aiosqlite not installed"
)


@pytest.fixture
async def sqlite_storage():
    """Create an in-memory SQLite storage for testing."""
    storage = SQLiteStorage(":memory:")
    yield storage


@pytest.fixture
async def temp_sqlite_storage():
    """Create a temporary file-based SQLite storage."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    storage = SQLiteStorage(db_path)
    yield storage

    # Cleanup
    try:
        os.unlink(db_path)
    except Exception:
        pass


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
async def test_save_and_retrieve_agent(sqlite_storage, sample_agent_record):
    """Test saving and retrieving an agent."""
    # Save agent
    await sqlite_storage.save_agent(sample_agent_record)

    # Retrieve agent
    retrieved = await sqlite_storage.get_agent(sample_agent_record.identity.id)

    assert retrieved is not None
    assert retrieved.identity.id == sample_agent_record.identity.id
    assert retrieved.identity.name == sample_agent_record.identity.name
    assert retrieved.identity.agent_type == sample_agent_record.identity.agent_type
    assert len(retrieved.capabilities) == len(sample_agent_record.capabilities)
    assert retrieved.metadata["version"] == "1.0"


@pytest.mark.asyncio
async def test_get_nonexistent_agent(sqlite_storage):
    """Test retrieving a non-existent agent returns None."""
    result = await sqlite_storage.get_agent("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_update_agent(sqlite_storage, sample_agent_record):
    """Test updating an existing agent."""
    # Save initial agent
    await sqlite_storage.save_agent(sample_agent_record)

    # Update agent metadata
    sample_agent_record.metadata["version"] = "2.0"

    # Save updated agent
    await sqlite_storage.save_agent(sample_agent_record)

    # Retrieve and verify
    retrieved = await sqlite_storage.get_agent(sample_agent_record.identity.id)

    assert retrieved.metadata["version"] == "2.0"


@pytest.mark.asyncio
async def test_list_all_agents(sqlite_storage):
    """Test listing all agents."""
    # Initially empty
    agents = await sqlite_storage.list_all()
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
        await sqlite_storage.save_agent(record)

    # List all
    agents = await sqlite_storage.list_all()
    assert len(agents) == 3
    assert {agent.identity.name for agent in agents} == {"agent-0", "agent-1", "agent-2"}


@pytest.mark.asyncio
async def test_search_with_fts(sqlite_storage):
    """Test full-text search functionality."""
    # Create agents with different capabilities
    agents_data = [
        ("summarizer", "Summarizes long texts", ["text-summarization"]),
        ("translator", "Translates between languages", ["translation"]),
        ("analyzer", "Analyzes data and statistics", ["data-analysis"]),
    ]

    for name, description, cap_names in agents_data:
        identity = AgentIdentity.create_simple(
            name=name,
            agent_type=AgentType.SOFTWARE,
            description=description,
        )
        capabilities = [
            Capability.create_simple(name=cap, description=f"{cap} capability")
            for cap in cap_names
        ]
        record = AgentRecord(
            identity=identity,
            capabilities=capabilities,
            registered_at=datetime.now(),
        )
        await sqlite_storage.save_agent(record)

    # Search for "translation"
    results = await sqlite_storage.search("translation", limit=10)
    assert len(results) >= 1
    assert any(agent.identity.name == "translator" for agent in results)

    # Search for "text"
    results = await sqlite_storage.search("text", limit=10)
    assert len(results) >= 1

    # Search for "data"
    results = await sqlite_storage.search("data", limit=10)
    assert len(results) >= 1
    assert any(agent.identity.name == "analyzer" for agent in results)


@pytest.mark.asyncio
async def test_search_with_limit(sqlite_storage):
    """Test search with result limit."""
    # Add 10 agents
    for i in range(10):
        identity = AgentIdentity.create_simple(
            name=f"agent-{i}",
            agent_type=AgentType.SOFTWARE,
            description="test agent",
        )
        record = AgentRecord(
            identity=identity,
            capabilities=[],
            registered_at=datetime.now(),
        )
        await sqlite_storage.save_agent(record)

    # Search with limit
    results = await sqlite_storage.search("agent", limit=5)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_search_with_capability_filter(sqlite_storage):
    """Test search with capability filters."""
    # Create agents with different capabilities
    identity1 = AgentIdentity.create_simple(
        name="agent1",
        agent_type=AgentType.SOFTWARE,
    )
    record1 = AgentRecord(
        identity=identity1,
        capabilities=[
            Capability.create_simple(name="nlp", description="NLP capability")
        ],
        registered_at=datetime.now(),
    )

    identity2 = AgentIdentity.create_simple(
        name="agent2",
        agent_type=AgentType.SOFTWARE,
    )
    record2 = AgentRecord(
        identity=identity2,
        capabilities=[
            Capability.create_simple(name="vision", description="Vision capability")
        ],
        registered_at=datetime.now(),
    )

    await sqlite_storage.save_agent(record1)
    await sqlite_storage.save_agent(record2)

    # Search with capability filter (use exact name match for FTS)
    results = await sqlite_storage.search(
        "agent1", limit=10, filters={"capabilities": ["nlp"]}
    )
    assert len(results) == 1
    assert results[0].identity.name == "agent1"


@pytest.mark.asyncio
async def test_search_with_tag_filter(sqlite_storage):
    """Test search with tag filters."""
    identity1 = AgentIdentity.create_simple(
        name="agent1",
        agent_type=AgentType.SOFTWARE,
    )
    record1 = AgentRecord(
        identity=identity1,
        capabilities=[
            Capability.create_simple(name="cap1", description="Cap 1", tags=["ai", "ml"])
        ],
        registered_at=datetime.now(),
    )

    identity2 = AgentIdentity.create_simple(
        name="agent2",
        agent_type=AgentType.SOFTWARE,
    )
    record2 = AgentRecord(
        identity=identity2,
        capabilities=[
            Capability.create_simple(
                name="cap2", description="Cap 2", tags=["data", "analytics"]
            )
        ],
        registered_at=datetime.now(),
    )

    await sqlite_storage.save_agent(record1)
    await sqlite_storage.save_agent(record2)

    # Search with tag filter (use exact name match for FTS)
    results = await sqlite_storage.search("agent1", limit=10, filters={"tags": ["ai"]})
    assert len(results) == 1
    assert results[0].identity.name == "agent1"


@pytest.mark.asyncio
async def test_persistence_across_connections(temp_sqlite_storage):
    """Test that data persists across different connections."""
    db_path = temp_sqlite_storage.db_path

    # Save agent with first connection
    identity = AgentIdentity.create_simple(
        name="persistent-agent",
        agent_type=AgentType.SOFTWARE,
    )
    record = AgentRecord(
        identity=identity,
        capabilities=[],
        registered_at=datetime.now(),
    )
    await temp_sqlite_storage.save_agent(record)

    # Create new connection to same database
    new_storage = SQLiteStorage(db_path)

    # Retrieve agent with new connection
    retrieved = await new_storage.get_agent(record.identity.id)
    assert retrieved is not None
    assert retrieved.identity.name == "persistent-agent"


@pytest.mark.asyncio
async def test_empty_capabilities(sqlite_storage):
    """Test saving and retrieving agent with no capabilities."""
    identity = AgentIdentity.create_simple(
        name="empty-agent",
        agent_type=AgentType.SOFTWARE,
    )
    record = AgentRecord(
        identity=identity,
        capabilities=[],
        registered_at=datetime.now(),
    )

    await sqlite_storage.save_agent(record)
    retrieved = await sqlite_storage.get_agent(record.identity.id)

    assert retrieved is not None
    assert retrieved.identity.name == "empty-agent"
    assert len(retrieved.capabilities) == 0


@pytest.mark.asyncio
async def test_complex_metadata(sqlite_storage, sample_agent_record):
    """Test saving and retrieving complex metadata."""
    sample_agent_record.metadata = {
        "version": "1.0",
        "config": {
            "timeout": 30,
            "retries": 3,
            "options": ["opt1", "opt2"],
        },
        "tags": ["production", "critical"],
    }

    await sqlite_storage.save_agent(sample_agent_record)
    retrieved = await sqlite_storage.get_agent(sample_agent_record.identity.id)

    assert retrieved.metadata["version"] == "1.0"
    assert retrieved.metadata["config"]["timeout"] == 30
    assert retrieved.metadata["tags"] == ["production", "critical"]


@pytest.mark.asyncio
async def test_capability_details_preserved(sqlite_storage):
    """Test that capability details are preserved."""
    identity = AgentIdentity.create_simple(
        name="detailed-agent",
        agent_type=AgentType.SOFTWARE,
    )

    cap = Capability.create_simple(
        name="detailed-cap",
        description="A detailed capability",
        tags=["tag1", "tag2", "tag3"],
    )

    record = AgentRecord(
        identity=identity,
        capabilities=[cap],
        registered_at=datetime.now(),
    )

    await sqlite_storage.save_agent(record)
    retrieved = await sqlite_storage.get_agent(record.identity.id)

    assert len(retrieved.capabilities) == 1
    retrieved_cap = retrieved.capabilities[0]
    assert retrieved_cap.name == "detailed-cap"
    assert retrieved_cap.description == "A detailed capability"
    if retrieved_cap.semantic:
        assert set(retrieved_cap.semantic.tags) == {"tag1", "tag2", "tag3"}


@pytest.mark.asyncio
async def test_search_no_results(sqlite_storage, sample_agent_record):
    """Test search with no matching results."""
    await sqlite_storage.save_agent(sample_agent_record)

    results = await sqlite_storage.search("nonexistent-query-xyz", limit=10)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_multiple_capabilities_same_name(sqlite_storage):
    """Test agent with multiple capabilities with different details."""
    identity = AgentIdentity.create_simple(
        name="multi-cap-agent",
        agent_type=AgentType.SOFTWARE,
    )

    capabilities = [
        Capability.create_simple(
            name="processing", description="Text processing", tags=["text"]
        ),
        Capability.create_simple(
            name="processing", description="Image processing", tags=["image"]
        ),
        Capability.create_simple(
            name="processing", description="Audio processing", tags=["audio"]
        ),
    ]

    record = AgentRecord(
        identity=identity,
        capabilities=capabilities,
        registered_at=datetime.now(),
    )

    await sqlite_storage.save_agent(record)
    retrieved = await sqlite_storage.get_agent(record.identity.id)

    assert len(retrieved.capabilities) == 3
    descriptions = {cap.description for cap in retrieved.capabilities}
    assert descriptions == {"Text processing", "Image processing", "Audio processing"}
