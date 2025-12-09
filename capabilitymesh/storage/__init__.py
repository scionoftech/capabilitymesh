"""Storage backends for CapabilityMesh.

This module provides pluggable storage for agent registry:
- InMemoryStorage: Zero-config, fast in-memory storage (default)
- SQLiteStorage: Local persistent storage
- RedisStorage: Distributed storage (coming soon)
"""

from .base import AgentRecord, Storage
from .memory import InMemoryStorage

__all__ = ["AgentRecord", "Storage", "InMemoryStorage"]

# Optional storage backends (require extra dependencies)
try:
    from .sqlite import SQLiteStorage

    __all__.append("SQLiteStorage")
except ImportError:
    pass

try:
    from .redis import RedisStorage

    __all__.append("RedisStorage")
except ImportError:
    pass
