"""Simple trust management system for CapabilityMesh.

Provides basic trust tracking based on execution results:
- TrustLevel: Enum for trust levels
- TrustScore: Dataclass for trust metrics
- SimpleTrustManager: Manager for tracking and updating trust scores
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Dict, List, Optional

from ..storage.base import Storage


class TrustLevel(IntEnum):
    """Trust levels for agents.

    Levels are ordered from least to most trusted:
    - UNTRUSTED: Never executed or explicitly untrusted
    - LOW: < 50% success rate or < 5 executions
    - MEDIUM: 50-80% success rate, >= 5 executions
    - HIGH: 80-95% success rate, >= 10 executions
    - VERIFIED: Manually verified or > 95% with >= 20 executions
    """

    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4


@dataclass
class TrustScore:
    """Trust score for an agent.

    Tracks execution statistics and trust level for an agent.
    """

    agent_id: str
    level: TrustLevel = TrustLevel.UNTRUSTED
    success_count: int = 0
    failure_count: int = 0
    total_executions: int = 0
    last_execution: Optional[datetime] = None
    manually_set: bool = False  # Whether trust level was manually set
    metadata: Dict[str, any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate.

        Returns:
            Success rate as a float between 0.0 and 1.0
        """
        if self.total_executions == 0:
            return 0.0
        return self.success_count / self.total_executions

    def auto_adjust_level(self) -> TrustLevel:
        """Auto-adjust trust level based on success rate and execution count.

        Returns:
            Recommended trust level based on statistics
        """
        if self.manually_set:
            # Don't auto-adjust if manually set
            return self.level

        if self.total_executions < 5:
            return TrustLevel.LOW

        rate = self.success_rate

        if rate >= 0.95 and self.total_executions >= 20:
            return TrustLevel.VERIFIED
        elif rate >= 0.80 and self.total_executions >= 10:
            return TrustLevel.HIGH
        elif rate >= 0.50:
            return TrustLevel.MEDIUM
        else:
            return TrustLevel.LOW


class SimpleTrustManager:
    """Simple trust manager for tracking agent reliability.

    Features:
    - Manual trust level setting
    - Automatic execution tracking
    - Success rate calculation
    - Trust-based filtering

    Example:
        trust = SimpleTrustManager()
        await trust.set_level(agent_id, TrustLevel.HIGH)
        await trust.record_execution(agent_id, success=True)
        score = await trust.get_score(agent_id)
    """

    def __init__(self, storage: Optional[Storage] = None):
        """Initialize trust manager.

        Args:
            storage: Optional storage backend for persistence
        """
        self.storage = storage
        self._trust_cache: Dict[str, TrustScore] = {}

    async def set_level(
        self, agent_id: str, level: TrustLevel, reason: Optional[str] = None
    ) -> None:
        """Manually set trust level for an agent.

        Args:
            agent_id: Agent ID
            level: Trust level to set
            reason: Optional reason for setting this level
        """
        # Get or create trust score
        score = await self.get_score(agent_id)

        # Update level and mark as manually set
        score.level = level
        score.manually_set = True

        if reason:
            score.metadata["manual_reason"] = reason
            score.metadata["manual_set_at"] = datetime.now().isoformat()

        # Save to cache
        self._trust_cache[agent_id] = score

        # TODO: Persist to storage if available
        # if self.storage:
        #     await self._persist_score(score)

    async def get_score(self, agent_id: str) -> TrustScore:
        """Get trust score for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            TrustScore for the agent
        """
        # Check cache first
        if agent_id in self._trust_cache:
            return self._trust_cache[agent_id]

        # TODO: Load from storage if available
        # if self.storage:
        #     score = await self._load_score(agent_id)
        #     if score:
        #         self._trust_cache[agent_id] = score
        #         return score

        # Create new score with UNTRUSTED level
        score = TrustScore(agent_id=agent_id, level=TrustLevel.UNTRUSTED)
        self._trust_cache[agent_id] = score
        return score

    async def record_execution(
        self, agent_id: str, success: bool, duration: Optional[float] = None
    ) -> None:
        """Record an execution result and update trust.

        Args:
            agent_id: Agent ID
            success: Whether execution was successful
            duration: Optional execution duration in seconds
        """
        # Get or create trust score
        score = await self.get_score(agent_id)

        # Update statistics
        score.total_executions += 1
        if success:
            score.success_count += 1
        else:
            score.failure_count += 1

        score.last_execution = datetime.now()

        if duration is not None:
            score.metadata["last_duration"] = duration

        # Auto-adjust level if not manually set
        if not score.manually_set:
            score.level = score.auto_adjust_level()

        # Save to cache
        self._trust_cache[agent_id] = score

        # TODO: Persist to storage if available
        # if self.storage:
        #     await self._persist_score(score)

    async def list_trusted_agents(
        self, min_level: TrustLevel = TrustLevel.MEDIUM
    ) -> List[TrustScore]:
        """List agents above a trust threshold.

        Args:
            min_level: Minimum trust level

        Returns:
            List of trust scores for agents at or above the threshold
        """
        results = []
        for score in self._trust_cache.values():
            if score.level >= min_level:
                results.append(score)

        # Sort by trust level (descending) then success rate (descending)
        results.sort(key=lambda s: (s.level, s.success_rate), reverse=True)

        return results

    async def get_stats(self) -> Dict[str, any]:
        """Get overall trust statistics.

        Returns:
            Dictionary with trust statistics
        """
        total_agents = len(self._trust_cache)
        level_counts = {level: 0 for level in TrustLevel}

        total_executions = 0
        total_successes = 0

        for score in self._trust_cache.values():
            level_counts[score.level] += 1
            total_executions += score.total_executions
            total_successes += score.success_count

        overall_success_rate = (
            total_successes / total_executions if total_executions > 0 else 0.0
        )

        return {
            "total_agents": total_agents,
            "level_distribution": {
                level.name: count for level, count in level_counts.items()
            },
            "total_executions": total_executions,
            "total_successes": total_successes,
            "overall_success_rate": overall_success_rate,
        }

    async def reset_agent(self, agent_id: str) -> bool:
        """Reset trust score for an agent.

        Args:
            agent_id: Agent ID to reset

        Returns:
            True if reset, False if agent not found
        """
        if agent_id in self._trust_cache:
            del self._trust_cache[agent_id]
            # TODO: Delete from storage if available
            return True
        return False

    async def clear_all(self) -> None:
        """Clear all trust scores.

        Useful for testing or resetting the system.
        """
        self._trust_cache.clear()
        # TODO: Clear from storage if available
