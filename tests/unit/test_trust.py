"""Unit tests for SimpleTrustManager.

Tests trust tracking, auto-adjustment, and integration with Mesh.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from capabilitymesh.trust import SimpleTrustManager, TrustLevel, TrustScore
from capabilitymesh.core.identity import AgentIdentity
from capabilitymesh.core.types import AgentType
from capabilitymesh.schemas.capability import Capability
from capabilitymesh.storage.base import AgentRecord
from capabilitymesh.storage.memory import InMemoryStorage
from capabilitymesh.mesh import Mesh


@pytest.fixture
def trust_manager():
    """Create a SimpleTrustManager instance for testing."""
    return SimpleTrustManager()


@pytest.fixture
def trust_manager_with_storage():
    """Create a SimpleTrustManager with storage backend."""
    storage = InMemoryStorage()
    return SimpleTrustManager(storage=storage)


@pytest.mark.asyncio
async def test_initial_trust_score(trust_manager):
    """Test that new agents start with UNTRUSTED level."""
    score = await trust_manager.get_score("agent1")

    assert score.agent_id == "agent1"
    assert score.level == TrustLevel.UNTRUSTED
    assert score.success_count == 0
    assert score.failure_count == 0
    assert score.total_executions == 0
    assert score.success_rate == 0.0


@pytest.mark.asyncio
async def test_manual_trust_level_setting(trust_manager):
    """Test manually setting trust level."""
    await trust_manager.set_level("agent1", TrustLevel.HIGH, reason="Verified by admin")

    score = await trust_manager.get_score("agent1")
    assert score.level == TrustLevel.HIGH
    assert score.manually_set is True
    assert score.metadata["manual_reason"] == "Verified by admin"
    assert "manual_set_at" in score.metadata


@pytest.mark.asyncio
async def test_record_successful_execution(trust_manager):
    """Test recording successful executions."""
    # Record 5 successful executions
    for _ in range(5):
        await trust_manager.record_execution("agent1", success=True, duration=0.5)

    score = await trust_manager.get_score("agent1")
    assert score.total_executions == 5
    assert score.success_count == 5
    assert score.failure_count == 0
    assert score.success_rate == 1.0
    assert score.metadata["last_duration"] == 0.5


@pytest.mark.asyncio
async def test_record_failed_execution(trust_manager):
    """Test recording failed executions."""
    # Record 3 successes and 2 failures
    for _ in range(3):
        await trust_manager.record_execution("agent1", success=True)
    for _ in range(2):
        await trust_manager.record_execution("agent1", success=False)

    score = await trust_manager.get_score("agent1")
    assert score.total_executions == 5
    assert score.success_count == 3
    assert score.failure_count == 2
    assert score.success_rate == 0.6


@pytest.mark.asyncio
async def test_auto_adjust_to_low(trust_manager):
    """Test auto-adjustment to LOW level."""
    # Record 3 successes (< 5 executions)
    for _ in range(3):
        await trust_manager.record_execution("agent1", success=True)

    score = await trust_manager.get_score("agent1")
    assert score.level == TrustLevel.LOW  # < 5 executions = LOW


@pytest.mark.asyncio
async def test_auto_adjust_to_medium(trust_manager):
    """Test auto-adjustment to MEDIUM level."""
    # Record 6 successes, 2 failures (60% success rate, >= 5 executions)
    for _ in range(6):
        await trust_manager.record_execution("agent1", success=True)
    for _ in range(2):
        await trust_manager.record_execution("agent1", success=False)

    score = await trust_manager.get_score("agent1")
    assert score.total_executions == 8
    assert 0.5 <= score.success_rate < 0.8  # 75% success rate
    assert score.level == TrustLevel.MEDIUM


@pytest.mark.asyncio
async def test_auto_adjust_to_high(trust_manager):
    """Test auto-adjustment to HIGH level."""
    # Record 17 successes, 2 failures (85% success rate, >= 10 executions)
    for _ in range(17):
        await trust_manager.record_execution("agent1", success=True)
    for _ in range(2):
        await trust_manager.record_execution("agent1", success=False)

    score = await trust_manager.get_score("agent1")
    assert score.total_executions == 19
    assert 0.80 <= score.success_rate < 0.95  # ~89.5% success rate
    assert score.level == TrustLevel.HIGH


@pytest.mark.asyncio
async def test_auto_adjust_to_verified(trust_manager):
    """Test auto-adjustment to VERIFIED level."""
    # Record 20 successes, 0 failures (100% success rate, >= 20 executions)
    for _ in range(20):
        await trust_manager.record_execution("agent1", success=True)

    score = await trust_manager.get_score("agent1")
    assert score.total_executions == 20
    assert score.success_rate >= 0.95
    assert score.level == TrustLevel.VERIFIED


@pytest.mark.asyncio
async def test_manual_level_not_auto_adjusted(trust_manager):
    """Test that manually set levels are not auto-adjusted."""
    # Manually set to HIGH
    await trust_manager.set_level("agent1", TrustLevel.HIGH)

    # Record failures that would normally drop to LOW
    for _ in range(10):
        await trust_manager.record_execution("agent1", success=False)

    score = await trust_manager.get_score("agent1")
    assert score.success_rate == 0.0  # All failures
    assert score.level == TrustLevel.HIGH  # Still HIGH (manually set)
    assert score.manually_set is True


@pytest.mark.asyncio
async def test_list_trusted_agents(trust_manager):
    """Test listing agents above a trust threshold."""
    # Create agents with different trust levels

    # Agent 1: HIGH (17 successes, 2 failures)
    for _ in range(17):
        await trust_manager.record_execution("agent1", success=True)
    for _ in range(2):
        await trust_manager.record_execution("agent1", success=False)

    # Agent 2: MEDIUM (6 successes, 2 failures)
    for _ in range(6):
        await trust_manager.record_execution("agent2", success=True)
    for _ in range(2):
        await trust_manager.record_execution("agent2", success=False)

    # Agent 3: LOW (2 successes)
    for _ in range(2):
        await trust_manager.record_execution("agent3", success=True)

    # Agent 4: VERIFIED (manually set)
    await trust_manager.set_level("agent4", TrustLevel.VERIFIED)

    # List agents with MEDIUM or above
    trusted = await trust_manager.list_trusted_agents(min_level=TrustLevel.MEDIUM)

    assert len(trusted) == 3  # agent1, agent2, agent4
    agent_ids = [score.agent_id for score in trusted]
    assert "agent1" in agent_ids
    assert "agent2" in agent_ids
    assert "agent4" in agent_ids
    assert "agent3" not in agent_ids

    # Verify sorting (by level descending, then success rate descending)
    assert trusted[0].level == TrustLevel.VERIFIED  # agent4


@pytest.mark.asyncio
async def test_get_stats(trust_manager):
    """Test getting overall trust statistics."""
    # Create agents with different levels

    # Agent 1: 10 successes (LOW level, < 5 would be better but we need >= 5 for MEDIUM)
    for _ in range(10):
        await trust_manager.record_execution("agent1", success=True)

    # Agent 2: 6 successes, 4 failures (MEDIUM level - 60%)
    for _ in range(6):
        await trust_manager.record_execution("agent2", success=True)
    for _ in range(4):
        await trust_manager.record_execution("agent2", success=False)

    # Agent 3: UNTRUSTED (no executions)
    await trust_manager.get_score("agent3")

    stats = await trust_manager.get_stats()

    assert stats["total_agents"] == 3
    assert stats["total_executions"] == 20  # 10 + 10
    assert stats["total_successes"] == 16  # 10 + 6
    assert stats["overall_success_rate"] == 0.8  # 16/20

    # Check level distribution
    level_dist = stats["level_distribution"]
    assert level_dist["UNTRUSTED"] == 1  # agent3
    assert level_dist["MEDIUM"] == 1  # agent2
    assert level_dist["HIGH"] == 1  # agent1 (10 successes, 100% rate, >= 10 executions)


@pytest.mark.asyncio
async def test_reset_agent(trust_manager):
    """Test resetting an agent's trust score."""
    # Create agent with some history
    for _ in range(5):
        await trust_manager.record_execution("agent1", success=True)

    score = await trust_manager.get_score("agent1")
    assert score.total_executions == 5

    # Reset agent
    result = await trust_manager.reset_agent("agent1")
    assert result is True

    # Score should be reset to initial state
    score = await trust_manager.get_score("agent1")
    assert score.level == TrustLevel.UNTRUSTED
    assert score.total_executions == 0


@pytest.mark.asyncio
async def test_reset_nonexistent_agent(trust_manager):
    """Test resetting a non-existent agent returns False."""
    result = await trust_manager.reset_agent("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_clear_all(trust_manager):
    """Test clearing all trust scores."""
    # Create multiple agents
    await trust_manager.record_execution("agent1", success=True)
    await trust_manager.record_execution("agent2", success=True)
    await trust_manager.record_execution("agent3", success=True)

    stats = await trust_manager.get_stats()
    assert stats["total_agents"] == 3

    # Clear all
    await trust_manager.clear_all()

    stats = await trust_manager.get_stats()
    assert stats["total_agents"] == 0


@pytest.mark.asyncio
async def test_trust_score_with_storage(trust_manager_with_storage):
    """Test that trust manager works with a storage backend."""
    # Record executions
    for _ in range(5):
        await trust_manager_with_storage.record_execution("agent1", success=True)

    score = await trust_manager_with_storage.get_score("agent1")
    assert score.total_executions == 5
    assert score.success_rate == 1.0


# Integration tests with Mesh


@pytest.fixture
async def mesh_with_trust():
    """Create a Mesh instance with trust manager for integration testing."""
    storage = InMemoryStorage()
    trust_manager = SimpleTrustManager(storage=storage)
    mesh = Mesh(storage=storage, trust_manager=trust_manager)

    # Define a simple function
    def test_function(task):
        if task == "fail":
            raise ValueError("Test failure")
        return f"Result: {task}"

    # Register the function as an agent
    capabilities = [
        Capability.create_simple(
            name="test-capability",
            description="A test capability",
            tags=["test"],
        )
    ]

    identity = await mesh.register(
        agent=test_function,
        name="test-agent",
        capabilities=capabilities,
        agent_type=AgentType.SOFTWARE,
        metadata={"description": "A test agent"},
    )

    return mesh, identity.id


@pytest.mark.asyncio
async def test_mesh_execute_tracks_success(mesh_with_trust):
    """Test that Mesh.execute() tracks successful executions."""
    mesh, agent_id = mesh_with_trust

    # Execute successfully
    result = await mesh.execute(agent_id, "test task")
    assert result == "Result: test task"

    # Check trust was updated
    score = await mesh.trust.get_score(agent_id)
    assert score.total_executions == 1
    assert score.success_count == 1
    assert score.failure_count == 0


@pytest.mark.asyncio
async def test_mesh_execute_tracks_failure(mesh_with_trust):
    """Test that Mesh.execute() tracks failed executions."""
    mesh, agent_id = mesh_with_trust

    # Execute and fail
    with pytest.raises(Exception):
        await mesh.execute(agent_id, "fail")

    # Check trust was updated
    score = await mesh.trust.get_score(agent_id)
    assert score.total_executions == 1
    assert score.success_count == 0
    assert score.failure_count == 1


@pytest.mark.asyncio
async def test_mesh_execute_tracks_duration(mesh_with_trust):
    """Test that Mesh.execute() tracks execution duration."""
    mesh, agent_id = mesh_with_trust

    # Execute successfully
    await mesh.execute(agent_id, "test task")

    # Check duration was tracked
    score = await mesh.trust.get_score(agent_id)
    assert "last_duration" in score.metadata
    assert score.metadata["last_duration"] >= 0  # Duration can be very small


@pytest.mark.asyncio
async def test_mesh_discover_with_trust_filter(mesh_with_trust):
    """Test that Mesh.discover() filters by trust level."""
    mesh, agent_id = mesh_with_trust

    # Initially, agent has UNTRUSTED level
    results = await mesh.discover("test", min_trust=TrustLevel.MEDIUM)
    assert len(results) == 0  # Filtered out

    # Execute successfully multiple times to raise trust to MEDIUM
    for _ in range(8):
        await mesh.execute(agent_id, "test task")

    # Now agent should appear
    results = await mesh.discover("test", min_trust=TrustLevel.MEDIUM)
    assert len(results) == 1
    assert results[0].id == agent_id


@pytest.mark.asyncio
async def test_mesh_discover_without_trust_filter(mesh_with_trust):
    """Test that Mesh.discover() works without trust filtering."""
    mesh, agent_id = mesh_with_trust

    # Without trust filter, agent should appear even if UNTRUSTED
    results = await mesh.discover("test", min_trust=None)
    assert len(results) == 1
    assert results[0].id == agent_id


@pytest.mark.asyncio
async def test_trust_progression_through_executions(mesh_with_trust):
    """Test trust level progression through multiple executions."""
    mesh, agent_id = mesh_with_trust

    # Start: UNTRUSTED
    score = await mesh.trust.get_score(agent_id)
    assert score.level == TrustLevel.UNTRUSTED

    # After 3 successes: LOW (< 5 executions)
    for _ in range(3):
        await mesh.execute(agent_id, "test task")
    score = await mesh.trust.get_score(agent_id)
    assert score.level == TrustLevel.LOW

    # After 8 total successes: MEDIUM (>= 5 executions, 100% rate)
    for _ in range(5):
        await mesh.execute(agent_id, "test task")
    score = await mesh.trust.get_score(agent_id)
    # 8 executions with 100% success should be HIGH (>= 80%, >= 10 would be needed)
    # Actually with 8 executions and 100%, we're still < 10 so might be MEDIUM or HIGH
    # Let me check the logic: >= 5 executions, >= 50% = MEDIUM, >= 80% and >= 10 = HIGH
    # So 8 executions at 100% should be MEDIUM (need >= 10 for HIGH)
    assert score.level == TrustLevel.MEDIUM

    # After 12 total successes: HIGH (>= 10 executions, >= 80% rate)
    for _ in range(4):
        await mesh.execute(agent_id, "test task")
    score = await mesh.trust.get_score(agent_id)
    assert score.level == TrustLevel.HIGH

    # After 20 total successes: VERIFIED (>= 20 executions, >= 95% rate)
    for _ in range(8):
        await mesh.execute(agent_id, "test task")
    score = await mesh.trust.get_score(agent_id)
    assert score.level == TrustLevel.VERIFIED


@pytest.mark.asyncio
async def test_trust_level_comparison(trust_manager):
    """Test that TrustLevel enum supports comparison operators."""
    assert TrustLevel.LOW < TrustLevel.MEDIUM
    assert TrustLevel.MEDIUM < TrustLevel.HIGH
    assert TrustLevel.HIGH < TrustLevel.VERIFIED
    assert TrustLevel.VERIFIED > TrustLevel.UNTRUSTED

    assert TrustLevel.MEDIUM >= TrustLevel.MEDIUM
    assert TrustLevel.HIGH >= TrustLevel.MEDIUM
