"""Mesh - The central hub for CapabilityMesh.

This module provides the main entry point for the CapabilityMesh API.
"""

import asyncio
import inspect
import threading
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from .core.identity import AgentIdentity, AgentAddress
from .core.types import AgentType
from .core.exceptions import (
    DiscoveryError,
    RegistrationError,
    ExecutionError,
)
from .schemas.capability import (
    Capability,
    CapabilityType,
)
from .storage import Storage, InMemoryStorage, AgentRecord
from .integrations.base import FrameworkAdapter
from .embeddings import auto_select_embedder, Embedder
from .trust import SimpleTrustManager, TrustLevel


class AgentInfo:
    """Information about a registered agent."""

    def __init__(self, record: AgentRecord):
        self.id = record.identity.id
        self.name = record.identity.name
        self.agent_type = record.identity.agent_type
        self.capabilities = record.capabilities
        self.metadata = record.metadata
        self.registered_at = record.registered_at

    def __repr__(self) -> str:
        caps = ", ".join([cap.name for cap in self.capabilities])
        return f"AgentInfo(id={self.id[:8]}..., name={self.name}, capabilities=[{caps}])"


class Mesh:
    """Central hub for multi-agent capability discovery and coordination.

    The Mesh class provides a simple, intuitive API for registering agents,
    discovering capabilities, and coordinating multi-agent workflows.

    Example:
        ```python
        from capabilitymesh import Mesh

        mesh = Mesh()

        # Register agents
        @mesh.agent(capabilities=["summarization"])
        def summarizer(text: str) -> str:
            return summarize(text)

        # Discover and execute
        agents = await mesh.discover("summarize this article")
        result = await mesh.execute(agents[0].id, "Long article text...")
        ```
    """

    def __init__(
        self,
        storage: Optional[Storage] = None,
        embedder: Optional[Embedder] = None,
        trust_manager: Optional[SimpleTrustManager] = None,
    ):
        """Initialize the Mesh.

        Args:
            storage: Storage backend (defaults to InMemoryStorage)
            embedder: Embedder for semantic search (auto-selected if None)
            trust_manager: Trust manager for tracking agent reliability (auto-created if None)
        """
        self.storage = storage or InMemoryStorage()
        self.embedder = embedder or auto_select_embedder()
        self.trust = trust_manager or SimpleTrustManager(storage=self.storage)
        self._function_registry: Dict[str, Callable] = {}
        self._capability_embeddings: Dict[str, List[float]] = {}  # cap_id -> embedding

    async def register(
        self,
        agent: Any,
        name: Optional[str] = None,
        capabilities: Optional[List[Union[str, Capability]]] = None,
        agent_type: Optional[AgentType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentIdentity:
        """Register an agent with the mesh.

        Supports:
        - Framework agents (CrewAI, AutoGen, LangGraph, A2A) - auto-extracts capabilities
        - Python functions/callables
        - Manual capability specification (overrides auto-extraction)

        Args:
            agent: Agent to register (framework agent, function, or callable)
            name: Optional name (auto-generated if not provided)
            capabilities: Optional capabilities (auto-extracted if not provided)
            agent_type: Optional agent type (auto-detected if not provided)
            metadata: Optional metadata

        Returns:
            AgentIdentity of the registered agent

        Raises:
            RegistrationError: If registration fails
        """
        try:
            # Detect if this is a framework adapter
            if isinstance(agent, FrameworkAdapter):
                return await self._register_adapter(
                    agent, name, capabilities, metadata
                )

            # Check if this is a known framework agent (CrewAI, AutoGen, etc.)
            framework_adapter = self._try_wrap_framework_agent(agent, name)
            if framework_adapter:
                return await self._register_adapter(
                    framework_adapter, name, capabilities, metadata
                )

            # Register as a callable/function
            return await self._register_callable(
                agent, name, capabilities, agent_type, metadata
            )

        except Exception as e:
            raise RegistrationError(f"Failed to register agent: {e}") from e

    async def _register_adapter(
        self,
        adapter: FrameworkAdapter,
        name: Optional[str] = None,
        capabilities: Optional[List[Union[str, Capability]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentIdentity:
        """Register a framework adapter."""
        identity = adapter.agent_identity

        # Use provided capabilities or auto-extract
        if capabilities is None:
            caps = adapter.extract_capabilities()
        else:
            caps = self._normalize_capabilities(capabilities, identity.name)

        # Compute embeddings for capabilities
        await self._compute_capability_embeddings(caps)

        # Create and save record
        record = AgentRecord(
            identity=identity,
            capabilities=caps,
            native_agent=adapter.framework_agent,
            metadata=metadata or {},
        )

        await self.storage.save_agent(record)
        return identity

    async def _register_callable(
        self,
        func: Callable,
        name: Optional[str] = None,
        capabilities: Optional[List[Union[str, Capability]]] = None,
        agent_type: Optional[AgentType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentIdentity:
        """Register a Python callable as an agent."""
        # Create identity
        agent_name = name or getattr(func, "__name__", f"agent-{uuid4().hex[:8]}")
        doc = inspect.getdoc(func) or f"Agent: {agent_name}"
        identity = AgentIdentity.create_simple(
            name=agent_name,
            agent_type=agent_type or AgentType.SOFTWARE,
            description=doc,
        )

        # Create capabilities
        if capabilities is None:
            # Create a default capability from the function
            caps = [self._create_capability_from_callable(func, agent_name)]
        else:
            caps = self._normalize_capabilities(capabilities, agent_name)

        # Compute embeddings for capabilities
        await self._compute_capability_embeddings(caps)

        # Store the function for execution
        self._function_registry[identity.id] = func

        # Create and save record
        record = AgentRecord(
            identity=identity,
            capabilities=caps,
            native_agent=func,
            metadata=metadata or {},
        )

        await self.storage.save_agent(record)
        return identity

    def _try_wrap_framework_agent(
        self, agent: Any, name: Optional[str] = None
    ) -> Optional[FrameworkAdapter]:
        """Try to wrap a framework-specific agent."""
        # Try CrewAI
        try:
            from .integrations.crewai import ACDPCrewAIAgent

            # Check if it's a CrewAI agent (duck typing)
            if hasattr(agent, "role") and hasattr(agent, "goal"):
                return ACDPCrewAIAgent.wrap(agent, name=name)
        except ImportError:
            pass

        # Try AutoGen
        try:
            from .integrations.autogen import ACDPAutoGenAgent

            # Check if it's an AutoGen agent
            if hasattr(agent, "send") and hasattr(agent, "receive"):
                return ACDPAutoGenAgent.wrap(agent, name=name)
        except ImportError:
            pass

        # Try LangGraph
        try:
            from .integrations.langgraph import ACDPLangGraphAgent

            # Check if it's a LangGraph agent
            if hasattr(agent, "invoke") and hasattr(agent, "ainvoke"):
                return ACDPLangGraphAgent.wrap(agent, name=name)
        except ImportError:
            pass

        return None

    def _normalize_capabilities(
        self, capabilities: List[Union[str, Capability]], agent_name: str
    ) -> List[Capability]:
        """Normalize capabilities to Capability objects."""
        normalized = []
        for cap in capabilities:
            if isinstance(cap, Capability):
                normalized.append(cap)
            elif isinstance(cap, str):
                # Create a simple capability from string
                normalized.append(
                    Capability.create_simple(
                        name=cap,
                        description=f"{agent_name} - {cap}",
                        capability_type=CapabilityType.UNSTRUCTURED,
                        tags=[cap],
                    )
                )
        return normalized

    def _create_capability_from_callable(
        self, func: Callable, agent_name: str
    ) -> Capability:
        """Create a capability from a Python callable."""
        doc = inspect.getdoc(func) or f"Capability: {agent_name}"

        return Capability.create_simple(
            name=agent_name,
            description=doc,
            capability_type=CapabilityType.STRUCTURED,
            tags=[agent_name],
        )

    async def _compute_capability_embeddings(
        self, capabilities: List[Capability]
    ) -> None:
        """Compute and store embeddings for capabilities.

        Args:
            capabilities: List of capabilities to embed
        """
        # Prepare texts to embed
        texts = []
        cap_ids = []

        for cap in capabilities:
            # Combine name, description, and tags for better matching
            text_parts = [cap.name, cap.description]
            if cap.semantic and cap.semantic.tags:
                text_parts.extend(cap.semantic.tags)

            text = " ".join(text_parts)
            texts.append(text)
            cap_ids.append(cap.id)

        # Compute embeddings in batch
        if texts:
            results = await self.embedder.embed_batch(texts)
            for cap_id, result in zip(cap_ids, results):
                self._capability_embeddings[cap_id] = result.embedding

    def agent(
        self,
        name: Optional[str] = None,
        capabilities: Optional[List[Union[str, Capability]]] = None,
        agent_type: Optional[AgentType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Decorator for registering agents.

        The agent is registered IMMEDIATELY when the decorator is applied,
        not when the function is first called.

        Example:
            ```python
            @mesh.agent(name="summarizer", capabilities=["summarization"])
            def summarize(text: str) -> str:
                return summarize_text(text)

            # Agent is already registered and discoverable
            agents = await mesh.discover("summarization")  # Finds the agent
            ```

        Args:
            name: Optional agent name
            capabilities: Optional capabilities
            agent_type: Optional agent type
            metadata: Optional metadata

        Returns:
            The original function (unmodified)
        """

        def decorator(func: Callable) -> Callable:
            # Register immediately using a background thread to avoid event loop issues
            self._register_in_background(func, name, capabilities, agent_type, metadata)

            # Return the ORIGINAL function - no wrapper needed
            return func

        return decorator

    def _register_in_background(
        self,
        func: Callable,
        name: Optional[str] = None,
        capabilities: Optional[List[Union[str, Capability]]] = None,
        agent_type: Optional[AgentType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an agent in a background thread to avoid blocking.

        This allows decorators to register immediately without blocking
        module import or causing event loop conflicts.

        Args:
            func: The function to register
            name: Optional agent name
            capabilities: Optional capabilities
            agent_type: Optional agent type
            metadata: Optional metadata
        """
        def _run_registration():
            # Create a new event loop in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Run the registration
                loop.run_until_complete(
                    self.register(func, name, capabilities, agent_type, metadata)
                )
            finally:
                loop.close()

        # Run registration in a daemon thread so it doesn't block
        thread = threading.Thread(target=_run_registration, daemon=True)
        thread.start()
        # Wait for registration to complete to ensure immediate availability
        thread.join(timeout=5.0)  # 5 second timeout to prevent hanging

    async def discover(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.001,
        min_trust: Optional[TrustLevel] = None,
    ) -> List[AgentInfo]:
        """Discover agents matching the query using semantic search.

        Uses embeddings for semantic similarity matching. Provides much better
        results than pure keyword matching. Optionally filters by trust level.

        Args:
            query: Search query (natural language or keywords)
            limit: Maximum number of results
            filters: Optional filters (e.g., agent_type, capabilities)
            min_similarity: Minimum similarity score (0.0 to 1.0)
            min_trust: Optional minimum trust level (filters out agents below this level)

        Returns:
            List of matching AgentInfo objects, sorted by relevance

        Raises:
            DiscoveryError: If discovery fails
        """
        try:
            # Get all agents
            all_records = await self.storage.list_all()

            # Apply filters first
            if filters:
                filtered_records = []
                for record in all_records:
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

                    filtered_records.append(record)
                all_records = filtered_records

            # Embed the query
            query_result = await self.embedder.embed(query)
            query_embedding = query_result.embedding

            # Calculate similarity scores for all agents
            agent_scores = []
            for record in all_records:
                # Calculate max similarity across all capabilities
                max_similarity = 0.0
                for cap in record.capabilities:
                    if cap.id in self._capability_embeddings:
                        cap_embedding = self._capability_embeddings[cap.id]
                        similarity = self.embedder.cosine_similarity(
                            query_embedding, cap_embedding
                        )
                        max_similarity = max(max_similarity, similarity)

                # Only include if above threshold
                if max_similarity >= min_similarity:
                    agent_scores.append((record, max_similarity))

            # Sort by similarity (descending)
            agent_scores.sort(key=lambda x: x[1], reverse=True)

            # Apply trust filtering if specified
            if min_trust is not None:
                filtered_scores = []
                for record, score in agent_scores:
                    trust_score = await self.trust.get_score(record.identity.id)
                    if trust_score.level >= min_trust:
                        filtered_scores.append((record, score))
                agent_scores = filtered_scores

            # Take top N
            top_records = [record for record, score in agent_scores[:limit]]

            return [AgentInfo(record) for record in top_records]
        except Exception as e:
            raise DiscoveryError(f"Discovery failed: {e}") from e

    async def discover_by_capability(
        self, capability: str, limit: int = 5
    ) -> List[AgentInfo]:
        """Discover agents by specific capability name.

        Args:
            capability: Capability name to search for
            limit: Maximum number of results

        Returns:
            List of matching AgentInfo objects
        """
        return await self.discover(capability, limit)

    async def list_agents(self) -> List[AgentInfo]:
        """List all registered agents.

        Returns:
            List of all AgentInfo objects
        """
        records = await self.storage.list_all()
        return [AgentInfo(record) for record in records]

    async def execute(
        self, agent_id: str, task: Any, **kwargs: Any
    ) -> Any:
        """Execute a task using an agent with automatic trust tracking.

        Provides a unified interface for executing tasks across different
        agent frameworks and types. Automatically tracks execution success/failure
        for trust management.

        Args:
            agent_id: ID of the agent to execute
            task: Task specification (format depends on agent type)
            **kwargs: Additional arguments passed to the agent

        Returns:
            Task result (format depends on agent type)

        Raises:
            ExecutionError: If execution fails
        """
        import time

        start_time = time.time()
        success = False

        try:
            record = await self.storage.get_agent(agent_id)
            if not record:
                raise ExecutionError(f"Agent not found: {agent_id}")

            # Check if it's a registered function
            if agent_id in self._function_registry:
                func = self._function_registry[agent_id]
                if asyncio.iscoroutinefunction(func):
                    result = await func(task, **kwargs)
                else:
                    result = func(task, **kwargs)
            # Otherwise, it should be a framework adapter
            elif isinstance(record.native_agent, FrameworkAdapter):
                result = record.native_agent.execute_task({"task": task, **kwargs})
            # Try to call it if it's callable
            elif callable(record.native_agent):
                if asyncio.iscoroutinefunction(record.native_agent):
                    result = await record.native_agent(task, **kwargs)
                else:
                    result = record.native_agent(task, **kwargs)
            else:
                raise ExecutionError(f"Agent {agent_id} is not executable")

            success = True
            return result

        except Exception as e:
            success = False
            raise ExecutionError(f"Execution failed: {e}") from e

        finally:
            # Track execution result for trust management
            duration = time.time() - start_time
            await self.trust.record_execution(agent_id, success, duration)

    async def get_native_async(self, agent_id: str) -> Any:
        """Get the native framework agent (async).

        Allows direct access to the underlying framework-specific agent
        for advanced use cases.

        Args:
            agent_id: ID of the agent

        Returns:
            Native framework agent

        Raises:
            DiscoveryError: If agent not found
        """
        record = await self.storage.get_agent(agent_id)
        if not record:
            raise DiscoveryError(f"Agent not found: {agent_id}")

        return record.native_agent

    def get_native(self, agent_id: str) -> Any:
        """Get the native framework agent (sync).

        Allows direct access to the underlying framework-specific agent
        for advanced use cases.

        Args:
            agent_id: ID of the agent

        Returns:
            Native framework agent

        Raises:
            DiscoveryError: If agent not found
        """
        import nest_asyncio

        try:
            # Try to use existing event loop with nest_asyncio for nested calls
            loop = asyncio.get_running_loop()
            nest_asyncio.apply(loop)
            return loop.run_until_complete(self.get_native_async(agent_id))
        except RuntimeError:
            # No running loop, create a new one
            return asyncio.run(self.get_native_async(agent_id))

    # Sync wrappers for convenience

    def discover_sync(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[AgentInfo]:
        """Synchronous wrapper for discover().

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters

        Returns:
            List of matching AgentInfo objects
        """
        import nest_asyncio

        try:
            # Try to use existing event loop with nest_asyncio for nested calls
            loop = asyncio.get_running_loop()
            nest_asyncio.apply(loop)
            return loop.run_until_complete(self.discover(query, limit, filters))
        except RuntimeError:
            # No running loop, create a new one
            return asyncio.run(self.discover(query, limit, filters))

    def execute_sync(self, agent_id: str, task: Any, **kwargs: Any) -> Any:
        """Synchronous wrapper for execute().

        Args:
            agent_id: ID of the agent to execute
            task: Task specification
            **kwargs: Additional arguments

        Returns:
            Task result
        """
        import nest_asyncio

        try:
            # Try to use existing event loop with nest_asyncio for nested calls
            loop = asyncio.get_running_loop()
            nest_asyncio.apply(loop)
            return loop.run_until_complete(self.execute(agent_id, task, **kwargs))
        except RuntimeError:
            # No running loop, create a new one
            return asyncio.run(self.execute(agent_id, task, **kwargs))
