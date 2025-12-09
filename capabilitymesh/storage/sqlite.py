"""SQLite-based storage backend for CapabilityMesh.

Provides persistent storage using SQLite with:
- Single file database
- Full-text search support (FTS5)
- Transaction safety
- Async operations using aiosqlite
"""

import json
import sqlite3  # For exception handling
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import aiosqlite
except ImportError:
    raise ImportError(
        "aiosqlite is required for SQLiteStorage. "
        "Install it with: pip install capabilitymesh[sqlite]"
    )

from .base import AgentRecord, Storage


class SQLiteStorage(Storage):
    """SQLite-based storage backend.

    Features:
    - Local file-based persistence
    - Full-text search using FTS5
    - Transaction safety
    - Connection pooling via aiosqlite
    - Automatic schema creation

    Example:
        storage = SQLiteStorage("agents.db")
        mesh = Mesh(storage=storage)
    """

    def __init__(self, db_path: str = "capabilitymesh.db"):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for in-memory DB.
        """
        self.db_path = db_path
        self._initialized = False
        self._shared_conn: Optional[aiosqlite.Connection] = None  # For :memory: DBs

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get a database connection and initialize schema if needed.

        For in-memory databases, reuses the same connection to preserve data.
        For file-based databases, creates a new connection each time.
        """
        # For in-memory databases, reuse the connection
        if self.db_path == ":memory:":
            if self._shared_conn is None:
                self._shared_conn = await aiosqlite.connect(self.db_path)
                self._shared_conn.row_factory = aiosqlite.Row
                await self._initialize_schema(self._shared_conn)
                self._initialized = True
            return self._shared_conn

        # For file-based databases, create new connection each time
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row

        if not self._initialized:
            await self._initialize_schema(conn)
            self._initialized = True

        return conn

    async def _initialize_schema(self, conn: aiosqlite.Connection) -> None:
        """Create database schema if it doesn't exist."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                description TEXT,
                registered_at TEXT,
                metadata TEXT
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS capabilities (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                capability_data TEXT,
                FOREIGN KEY(agent_id) REFERENCES agents(id) ON DELETE CASCADE
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trust_scores (
                agent_id TEXT PRIMARY KEY,
                trust_level INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                total_executions INTEGER NOT NULL DEFAULT 0,
                last_execution TEXT,
                FOREIGN KEY(agent_id) REFERENCES agents(id) ON DELETE CASCADE
            )
        """)

        # Create FTS5 virtual table for full-text search
        await conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS agents_fts USING fts5(
                agent_id UNINDEXED,
                name,
                description,
                capabilities
            )
        """)

        # Create indexes for performance
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_capabilities_agent_id
            ON capabilities(agent_id)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_capabilities_name
            ON capabilities(name)
        """)

        await conn.commit()

    async def save_agent(self, record: AgentRecord) -> None:
        """Save an agent record to SQLite.

        Args:
            record: Agent record to save
        """
        conn = await self._get_connection()

        try:
            # Get agent_id from identity
            agent_id = record.identity.id

            # Serialize metadata
            metadata_json = json.dumps(record.metadata) if record.metadata else "{}"

            # Insert or replace agent
            await conn.execute(
                """
                INSERT OR REPLACE INTO agents
                (id, name, agent_type, description, registered_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    record.identity.name,
                    record.identity.agent_type.value,
                    getattr(record.identity, "description", "") or "",
                    record.registered_at.isoformat() if record.registered_at else None,
                    metadata_json,
                ),
            )

            # Delete existing capabilities
            await conn.execute(
                "DELETE FROM capabilities WHERE agent_id = ?", (agent_id,)
            )

            # Insert capabilities
            for cap in record.capabilities:
                # Extract tags and categories from semantic field
                tags = []
                categories = []
                if cap.semantic:
                    tags = cap.semantic.tags or []
                    categories = cap.semantic.categories or []

                cap_data = {
                    "type": cap.capability_type.value if cap.capability_type else "unstructured",
                    "tags": tags,
                    "categories": categories,
                }
                await conn.execute(
                    """
                    INSERT INTO capabilities
                    (id, agent_id, name, description, capability_data)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        cap.id,
                        agent_id,
                        cap.name,
                        cap.description or "",
                        json.dumps(cap_data),
                    ),
                )

            # Update FTS index
            capability_names = ", ".join([cap.name for cap in record.capabilities])
            await conn.execute(
                """
                INSERT OR REPLACE INTO agents_fts
                (agent_id, name, description, capabilities)
                VALUES (?, ?, ?, ?)
                """,
                (
                    agent_id,
                    record.identity.name,
                    getattr(record.identity, "description", "") or "",
                    capability_names,
                ),
            )

            await conn.commit()

        finally:
            # Don't close shared connection for in-memory databases
            if conn is not self._shared_conn:
                await conn.close()

    async def get_agent(self, agent_id: str) -> Optional[AgentRecord]:
        """Retrieve an agent by ID.

        Args:
            agent_id: Agent ID to retrieve

        Returns:
            AgentRecord if found, None otherwise
        """
        conn = await self._get_connection()

        try:
            # Get agent data
            cursor = await conn.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            )
            row = await cursor.fetchone()

            if not row:
                return None

            # Get capabilities
            cursor = await conn.execute(
                "SELECT * FROM capabilities WHERE agent_id = ?", (agent_id,)
            )
            cap_rows = await cursor.fetchall()

            # Build AgentRecord
            return await self._build_agent_record(row, cap_rows)

        finally:
            # Don't close shared connection for in-memory databases
            if conn is not self._shared_conn:
                await conn.close()

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[AgentRecord]:
        """Search agents using full-text search.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters (capabilities, tags, etc.)

        Returns:
            List of matching agent records
        """
        conn = await self._get_connection()

        try:
            # Use FTS5 for full-text search
            # Try exact search first, fallback to escaped search if it fails
            fts_rows = []
            try:
                cursor = await conn.execute(
                    """
                    SELECT agent_id FROM agents_fts
                    WHERE agents_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit),
                )
                fts_rows = await cursor.fetchall()
            except sqlite3.OperationalError:
                # If FTS query fails (e.g., due to special characters),
                # escape the query and try again
                escaped_query = '"' + query.replace('"', '""') + '"'
                try:
                    cursor = await conn.execute(
                        """
                        SELECT agent_id FROM agents_fts
                        WHERE agents_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (escaped_query, limit),
                    )
                    fts_rows = await cursor.fetchall()
                except sqlite3.OperationalError:
                    # If even escaped query fails, return empty results
                    fts_rows = []

            if not fts_rows:
                return []

            # Get full agent records
            agent_ids = [row["agent_id"] for row in fts_rows]
            placeholders = ",".join(["?"] * len(agent_ids))

            cursor = await conn.execute(
                f"SELECT * FROM agents WHERE id IN ({placeholders})", agent_ids
            )
            agent_rows = await cursor.fetchall()

            # Get capabilities for all agents
            cursor = await conn.execute(
                f"SELECT * FROM capabilities WHERE agent_id IN ({placeholders})",
                agent_ids,
            )
            all_cap_rows = await cursor.fetchall()

            # Group capabilities by agent_id
            capabilities_by_agent = {}
            for cap_row in all_cap_rows:
                agent_id = cap_row["agent_id"]
                if agent_id not in capabilities_by_agent:
                    capabilities_by_agent[agent_id] = []
                capabilities_by_agent[agent_id].append(cap_row)

            # Build agent records
            results = []
            for agent_row in agent_rows:
                agent_id = agent_row["id"]
                cap_rows = capabilities_by_agent.get(agent_id, [])
                record = await self._build_agent_record(agent_row, cap_rows)

                # Apply filters
                if filters and not self._matches_filters(record, filters):
                    continue

                results.append(record)

            return results[:limit]

        finally:
            # Don't close shared connection for in-memory databases
            if conn is not self._shared_conn:
                await conn.close()

    async def list_all(self) -> List[AgentRecord]:
        """List all agents in storage.

        Returns:
            List of all agent records
        """
        conn = await self._get_connection()

        try:
            # Get all agents
            cursor = await conn.execute("SELECT * FROM agents")
            agent_rows = await cursor.fetchall()

            if not agent_rows:
                return []

            # Get all capabilities
            cursor = await conn.execute("SELECT * FROM capabilities")
            all_cap_rows = await cursor.fetchall()

            # Group capabilities by agent_id
            capabilities_by_agent = {}
            for cap_row in all_cap_rows:
                agent_id = cap_row["agent_id"]
                if agent_id not in capabilities_by_agent:
                    capabilities_by_agent[agent_id] = []
                capabilities_by_agent[agent_id].append(cap_row)

            # Build agent records
            results = []
            for agent_row in agent_rows:
                agent_id = agent_row["id"]
                cap_rows = capabilities_by_agent.get(agent_id, [])
                record = await self._build_agent_record(agent_row, cap_rows)
                results.append(record)

            return results

        finally:
            # Don't close shared connection for in-memory databases
            if conn is not self._shared_conn:
                await conn.close()

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID.

        Args:
            agent_id: Agent ID to delete

        Returns:
            True if deleted, False if not found
        """
        conn = await self._get_connection()

        try:
            # Check if agent exists
            cursor = await conn.execute(
                "SELECT id FROM agents WHERE id = ?", (agent_id,)
            )
            row = await cursor.fetchone()

            if not row:
                return False

            # Delete agent (cascades to capabilities and trust_scores)
            await conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))

            # Delete from FTS index
            await conn.execute(
                "DELETE FROM agents_fts WHERE agent_id = ?", (agent_id,)
            )

            await conn.commit()
            return True

        finally:
            # Don't close shared connection for in-memory databases
            if conn is not self._shared_conn:
                await conn.close()

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
        conn = await self._get_connection()

        try:
            # Get current metadata
            cursor = await conn.execute(
                "SELECT metadata FROM agents WHERE id = ?", (agent_id,)
            )
            row = await cursor.fetchone()

            if not row:
                return False

            # Merge metadata
            current_metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            current_metadata.update(metadata)

            # Update metadata
            await conn.execute(
                "UPDATE agents SET metadata = ? WHERE id = ?",
                (json.dumps(current_metadata), agent_id),
            )

            await conn.commit()
            return True

        finally:
            # Don't close shared connection for in-memory databases
            if conn is not self._shared_conn:
                await conn.close()

    async def _build_agent_record(
        self, agent_row: aiosqlite.Row, cap_rows: List[aiosqlite.Row]
    ) -> AgentRecord:
        """Build an AgentRecord from database rows.

        Args:
            agent_row: Agent table row
            cap_rows: Capability table rows

        Returns:
            AgentRecord instance
        """
        from capabilitymesh.core.identity import AgentIdentity
        from capabilitymesh.core.types import AgentType
        from capabilitymesh.schemas.capability import Capability, CapabilityType

        # Parse metadata
        metadata = json.loads(agent_row["metadata"]) if agent_row["metadata"] else {}

        # Build identity
        identity = AgentIdentity.create_simple(
            name=agent_row["name"],
            agent_type=AgentType(agent_row["agent_type"]),
            description=agent_row["description"],
        )
        # Override DID with stored ID
        identity.did = agent_row["id"]

        # Build capabilities
        capabilities = []
        for cap_row in cap_rows:
            cap_data = json.loads(cap_row["capability_data"])
            cap = Capability.create_simple(
                name=cap_row["name"],
                description=cap_row["description"] or "",
                capability_type=CapabilityType(cap_data.get("type", "unstructured")),
                tags=cap_data.get("tags", []),
            )
            cap.id = cap_row["id"]

            # Update semantic categories if present
            if cap_data.get("categories") and cap.semantic:
                cap.semantic.categories = cap_data["categories"]

            capabilities.append(cap)

        # Parse registered_at
        registered_at = None
        if agent_row["registered_at"]:
            try:
                registered_at = datetime.fromisoformat(agent_row["registered_at"])
            except (ValueError, TypeError):
                pass

        return AgentRecord(
            identity=identity,
            capabilities=capabilities,
            metadata=metadata,
            registered_at=registered_at,
        )

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
