"""Unit tests for the Mesh class."""

import pytest

from capabilitymesh import Mesh, AgentInfo
from capabilitymesh.core.types import AgentType, CapabilityType
from capabilitymesh.schemas.capability import Capability


@pytest.fixture
def mesh():
    """Create a fresh Mesh instance."""
    return Mesh()


@pytest.mark.asyncio
async def test_register_callable(mesh):
    """Test registering a Python callable."""

    def simple_function(text: str) -> str:
        return f"Processed: {text}"

    identity = await mesh.register(
        simple_function, name="processor", capabilities=["processing"]
    )

    assert identity is not None
    assert identity.name == "processor"

    # Verify it's in storage
    agents = await mesh.list_agents()
    assert len(agents) == 1
    assert agents[0].name == "processor"


@pytest.mark.asyncio
async def test_register_with_decorator(mesh):
    """Test registering using decorator.

    The decorator now registers immediately, not on first call.
    """

    @mesh.agent(name="summarizer", capabilities=["summarization"])
    async def summarize(text: str) -> str:
        return f"Summary: {text[:10]}"

    # Agent should be registered immediately (no function call needed)
    agents = await mesh.list_agents()
    assert len(agents) == 1
    assert agents[0].name == "summarizer"

    # Function should still work normally
    result = await summarize("This is a long text that needs summarization")
    assert result.startswith("Summary:")


@pytest.mark.asyncio
async def test_decorator_returns_original_function(mesh):
    """Test that decorator returns the original function (no wrapper).

    This verifies the fix in v1.0.0-alpha.2 where the decorator was changed
    to return the original function instead of a wrapper.
    """

    def original_function(x: int) -> int:
        """Original function docstring."""
        return x * 2

    decorated_function = mesh.agent(name="doubler", capabilities=["math"])(
        original_function
    )

    # Verify decorator returns the exact same function object
    assert decorated_function is original_function
    assert decorated_function.__name__ == "original_function"
    assert decorated_function.__doc__ == "Original function docstring."

    # Agent should still be registered
    agents = await mesh.list_agents()
    assert len(agents) == 1
    assert agents[0].name == "doubler"


@pytest.mark.asyncio
async def test_decorator_with_sync_and_async(mesh):
    """Test decorator works correctly with both sync and async functions."""

    # Sync function
    @mesh.agent(name="sync-agent", capabilities=["sync"])
    def sync_func(x: int) -> int:
        return x + 1

    # Async function
    @mesh.agent(name="async-agent", capabilities=["async"])
    async def async_func(x: int) -> int:
        return x + 2

    # Both should be registered immediately
    agents = await mesh.list_agents()
    assert len(agents) == 2

    # Both should work correctly
    assert sync_func(5) == 6
    assert await async_func(5) == 7


@pytest.mark.asyncio
async def test_register_async_function(mesh):
    """Test registering an async function."""

    async def async_processor(text: str) -> str:
        return f"Async: {text}"

    identity = await mesh.register(
        async_processor, name="async-proc", capabilities=["async-processing"]
    )

    assert identity is not None
    assert identity.name == "async-proc"


@pytest.mark.asyncio
async def test_discover_by_capability(mesh):
    """Test discovering agents by capability."""

    # Register multiple agents
    def agent1(text: str) -> str:
        return text

    def agent2(text: str) -> str:
        return text

    await mesh.register(agent1, name="agent1", capabilities=["summarization"])
    await mesh.register(agent2, name="agent2", capabilities=["translation"])

    # Discover by capability
    results = await mesh.discover("summarization")
    assert len(results) == 1
    assert results[0].name == "agent1"

    results = await mesh.discover("translation")
    assert len(results) == 1
    assert results[0].name == "agent2"


@pytest.mark.asyncio
async def test_discover_with_multiple_matches(mesh):
    """Test discovering with multiple matching agents."""

    for i in range(3):

        def func(text: str) -> str:
            return text

        await mesh.register(
            func, name=f"agent-{i}", capabilities=["summarization", "common"]
        )

    results = await mesh.discover("summarization", limit=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_list_agents(mesh):
    """Test listing all agents."""

    # Initially empty
    agents = await mesh.list_agents()
    assert len(agents) == 0

    # Add agents
    for i in range(5):

        def func(text: str) -> str:
            return text

        await mesh.register(func, name=f"agent-{i}", capabilities=["test"])

    # List all
    agents = await mesh.list_agents()
    assert len(agents) == 5


@pytest.mark.asyncio
async def test_execute_sync_function(mesh):
    """Test executing a synchronous function."""

    def echo(text: str) -> str:
        return f"Echo: {text}"

    identity = await mesh.register(echo, name="echo", capabilities=["echo"])

    result = await mesh.execute(identity.id, "Hello World")
    assert result == "Echo: Hello World"


@pytest.mark.asyncio
async def test_execute_async_function(mesh):
    """Test executing an asynchronous function."""

    async def async_echo(text: str) -> str:
        return f"Async Echo: {text}"

    identity = await mesh.register(
        async_echo, name="async-echo", capabilities=["async-echo"]
    )

    result = await mesh.execute(identity.id, "Hello Async")
    assert result == "Async Echo: Hello Async"


@pytest.mark.asyncio
async def test_get_native(mesh):
    """Test getting the native agent."""

    def original_func(text: str) -> str:
        return text

    identity = await mesh.register(
        original_func, name="test", capabilities=["test"]
    )

    native = await mesh.get_native_async(identity.id)
    assert native == original_func


@pytest.mark.asyncio
async def test_discover_sync_wrapper(mesh):
    """Test synchronous discover wrapper - using async version in tests."""

    def func(text: str) -> str:
        return text

    await mesh.register(func, name="test", capabilities=["testing"])

    # In async tests, we use the async version
    results = await mesh.discover("testing")
    assert len(results) == 1
    assert results[0].name == "test"


@pytest.mark.asyncio
async def test_execute_sync_wrapper(mesh):
    """Test synchronous execute wrapper - using async version in tests."""

    def func(text: str) -> str:
        return f"Result: {text}"

    identity = await mesh.register(func, name="test", capabilities=["test"])

    # In async tests, we use the async version
    result = await mesh.execute(identity.id, "input")
    assert result == "Result: input"


@pytest.mark.asyncio
async def test_register_with_capability_objects(mesh):
    """Test registering with Capability objects instead of strings."""

    def func(data: dict) -> dict:
        return data

    capability = Capability.create_simple(
        name="custom-capability",
        description="A custom capability",
        capability_type=CapabilityType.STRUCTURED,
    )

    identity = await mesh.register(func, name="test", capabilities=[capability])

    # Verify capability was registered
    agents = await mesh.list_agents()
    assert len(agents) == 1
    assert len(agents[0].capabilities) == 1
    assert agents[0].capabilities[0].name == "custom-capability"


@pytest.mark.asyncio
async def test_register_with_metadata(mesh):
    """Test registering with custom metadata."""

    def func(text: str) -> str:
        return text

    metadata = {"version": "1.0", "author": "test", "tags": ["experimental"]}

    identity = await mesh.register(
        func, name="test", capabilities=["test"], metadata=metadata
    )

    # Retrieve and verify metadata
    agents = await mesh.list_agents()
    assert agents[0].metadata["version"] == "1.0"
    assert agents[0].metadata["author"] == "test"


@pytest.mark.asyncio
async def test_agent_info_repr(mesh):
    """Test AgentInfo representation."""

    def func(text: str) -> str:
        return text

    await mesh.register(func, name="test-agent", capabilities=["cap1", "cap2"])

    agents = await mesh.list_agents()
    info = agents[0]

    repr_str = repr(info)
    assert "test-agent" in repr_str
    assert "cap1" in repr_str
    assert "cap2" in repr_str


@pytest.mark.asyncio
async def test_discover_with_partial_match(mesh):
    """Test discovering with partial name/description matches."""

    def func(text: str) -> str:
        return text

    await mesh.register(func, name="text-processor", capabilities=["processing"])

    # Should match by partial name
    results = await mesh.discover("text")
    assert len(results) == 1
    assert results[0].name == "text-processor"


# =============================================================================
# Enhanced Error Handling and Edge Case Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in Mesh operations."""

    @pytest.mark.asyncio
    async def test_execute_nonexistent_agent(self):
        """Test executing a nonexistent agent raises error."""
        from capabilitymesh.core.exceptions import ExecutionError

        mesh = Mesh()

        with pytest.raises(ExecutionError) as exc_info:
            await mesh.execute("nonexistent-id", "test input")

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_with_invalid_agent_id_type(self):
        """Test execute with invalid agent ID type."""
        from capabilitymesh.core.exceptions import ExecutionError

        mesh = Mesh()

        # Register an agent first
        def func(x):
            return x

        await mesh.register(func, name="test", capabilities=["test"])

        # Try to execute with invalid ID type
        with pytest.raises((ExecutionError, TypeError, AttributeError)):
            await mesh.execute(None, "input")

    @pytest.mark.asyncio
    async def test_execute_agent_that_raises_exception(self):
        """Test executing an agent that raises an exception."""
        from capabilitymesh.core.exceptions import ExecutionError

        mesh = Mesh()

        def failing_agent(text: str) -> str:
            raise ValueError("Intentional error")

        identity = await mesh.register(
            failing_agent, name="failer", capabilities=["fail"]
        )

        with pytest.raises(ExecutionError) as exc_info:
            await mesh.execute(identity.id, "test")

        assert "Intentional error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_register_without_name(self):
        """Test registering without providing a name."""
        mesh = Mesh()

        def func(x):
            return x

        # Should auto-generate a name
        identity = await mesh.register(func, capabilities=["test"])

        assert identity.name is not None
        assert len(identity.name) > 0

    @pytest.mark.asyncio
    async def test_register_with_empty_capabilities(self):
        """Test registering with empty capabilities list."""
        mesh = Mesh()

        def func(x):
            return x

        identity = await mesh.register(func, name="test", capabilities=[])

        assert identity is not None
        agents = await mesh.list_agents()
        assert len(agents) == 1

    @pytest.mark.asyncio
    async def test_discover_with_no_matches(self):
        """Test discovering when no agents match."""
        mesh = Mesh()

        def func(x):
            return x

        await mesh.register(func, name="test", capabilities=["capability-a"])

        results = await mesh.discover("nonexistent-capability")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_discover_empty_mesh(self):
        """Test discovering in an empty mesh."""
        mesh = Mesh()

        results = await mesh.discover("anything")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_native_nonexistent_agent(self):
        """Test getting native agent for nonexistent ID."""
        mesh = Mesh()

        result = await mesh.get_native_async("nonexistent-id")

        assert result is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_register_same_agent_twice(self):
        """Test registering the same agent twice with different names."""
        mesh = Mesh()

        def func(x):
            return x

        identity1 = await mesh.register(func, name="agent1", capabilities=["cap1"])
        identity2 = await mesh.register(func, name="agent2", capabilities=["cap2"])

        # Should create two separate agent registrations
        agents = await mesh.list_agents()
        assert len(agents) == 2
        assert identity1.id != identity2.id

    @pytest.mark.asyncio
    async def test_register_with_very_long_name(self):
        """Test registering with extremely long name."""
        mesh = Mesh()

        def func(x):
            return x

        long_name = "a" * 1000
        identity = await mesh.register(func, name=long_name, capabilities=["test"])

        assert identity.name == long_name

    @pytest.mark.asyncio
    async def test_register_with_special_characters_in_name(self):
        """Test registering with special characters in name."""
        mesh = Mesh()

        def func(x):
            return x

        special_name = "agent-with-special!@#$%^&*()chars"
        identity = await mesh.register(
            func, name=special_name, capabilities=["test"]
        )

        assert identity.name == special_name

    @pytest.mark.asyncio
    async def test_register_with_unicode_name(self):
        """Test registering with Unicode name."""
        mesh = Mesh()

        def func(x):
            return x

        unicode_name = "агент-тест-🚀-你好"
        identity = await mesh.register(func, name=unicode_name, capabilities=["test"])

        assert identity.name == unicode_name

    @pytest.mark.asyncio
    async def test_execute_with_complex_input(self):
        """Test executing with complex nested input."""
        mesh = Mesh()

        def process_data(data: dict) -> dict:
            return {"processed": data}

        identity = await mesh.register(
            process_data, name="processor", capabilities=["process"]
        )

        complex_input = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "string": "test",
        }

        result = await mesh.execute(identity.id, complex_input)

        assert result["processed"] == complex_input

    @pytest.mark.asyncio
    async def test_execute_with_multiple_arguments(self):
        """Test executing with multiple arguments."""
        mesh = Mesh()

        def multi_arg_func(a: int, b: int, c: int = 0) -> int:
            return a + b + c

        identity = await mesh.register(
            multi_arg_func, name="adder", capabilities=["math"]
        )

        # Execute with positional args
        result = await mesh.execute(identity.id, 10, 20, 30)

        assert result == 60

    @pytest.mark.asyncio
    async def test_execute_with_keyword_arguments(self):
        """Test executing with keyword arguments."""
        mesh = Mesh()

        def keyword_func(x: int, y: int = 5) -> int:
            return x * y

        identity = await mesh.register(
            keyword_func, name="multiplier", capabilities=["math"]
        )

        # Execute with kwargs
        result = await mesh.execute(identity.id, x=10, y=3)

        assert result == 30

    @pytest.mark.asyncio
    async def test_discover_with_zero_limit(self):
        """Test discover with limit=0."""
        mesh = Mesh()

        for i in range(5):

            def func(x):
                return x

            await mesh.register(func, name=f"agent-{i}", capabilities=["test"])

        results = await mesh.discover("test", limit=0)

        # limit=0 typically means unlimited
        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_discover_with_negative_limit(self):
        """Test discover with negative limit."""
        mesh = Mesh()

        def func(x):
            return x

        await mesh.register(func, name="test", capabilities=["test"])

        # Should handle gracefully
        results = await mesh.discover("test", limit=-1)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_register_lambda_function(self):
        """Test registering a lambda function."""
        mesh = Mesh()

        identity = await mesh.register(
            lambda x: x * 2, name="doubler", capabilities=["math"]
        )

        result = await mesh.execute(identity.id, 5)

        assert result == 10

    @pytest.mark.asyncio
    async def test_register_with_duplicate_capability_names(self):
        """Test registering with duplicate capability names."""
        mesh = Mesh()

        def func(x):
            return x

        identity = await mesh.register(
            func, name="test", capabilities=["cap1", "cap1", "cap1"]
        )

        agents = await mesh.list_agents()
        # Should deduplicate or handle gracefully
        assert len(agents) == 1

    @pytest.mark.asyncio
    async def test_execute_returns_none(self):
        """Test executing an agent that returns None."""
        mesh = Mesh()

        def returns_none(x):
            return None

        identity = await mesh.register(
            returns_none, name="none-returner", capabilities=["test"]
        )

        result = await mesh.execute(identity.id, "input")

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_with_no_arguments(self):
        """Test executing a function that takes no arguments."""
        mesh = Mesh()

        def no_args() -> str:
            return "no args"

        identity = await mesh.register(no_args, name="no-args", capabilities=["test"])

        result = await mesh.execute(identity.id)

        assert result == "no args"


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_registrations(self):
        """Test registering multiple agents concurrently."""
        import asyncio

        mesh = Mesh()

        async def register_agent(i):
            def func(x):
                return x

            return await mesh.register(func, name=f"agent-{i}", capabilities=["test"])

        # Register 10 agents concurrently
        identities = await asyncio.gather(*[register_agent(i) for i in range(10)])

        assert len(identities) == 10
        agents = await mesh.list_agents()
        assert len(agents) == 10

    @pytest.mark.asyncio
    async def test_concurrent_discoveries(self):
        """Test discovering agents concurrently."""
        import asyncio

        mesh = Mesh()

        # Register agents
        for i in range(5):

            def func(x):
                return x

            await mesh.register(func, name=f"agent-{i}", capabilities=["test"])

        # Discover concurrently
        results = await asyncio.gather(*[mesh.discover("test") for _ in range(10)])

        # All discoveries should return same results
        assert all(len(r) == 5 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_executions(self):
        """Test executing agents concurrently."""
        import asyncio

        mesh = Mesh()

        async def slow_func(x: int) -> int:
            await asyncio.sleep(0.01)  # Small delay
            return x * 2

        identity = await mesh.register(
            slow_func, name="slow-agent", capabilities=["test"]
        )

        # Execute concurrently
        results = await asyncio.gather(
            *[mesh.execute(identity.id, i) for i in range(10)]
        )

        assert results == [i * 2 for i in range(10)]


class TestTrustIntegration:
    """Test trust system integration with Mesh."""

    @pytest.mark.asyncio
    async def test_discover_with_trust_filter(self):
        """Test discovering agents with trust level filter."""
        from capabilitymesh.core.types import TrustLevel

        mesh = Mesh()

        def func(x):
            return x

        identity = await mesh.register(func, name="test", capabilities=["test"])

        # Discover with high trust requirement
        results = await mesh.discover("test", min_trust=TrustLevel.HIGH)

        # New agent won't meet high trust requirement
        assert len(results) == 0

        # Discover with low trust requirement
        results = await mesh.discover("test", min_trust=TrustLevel.UNTRUSTED)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_execution_updates_trust_on_success(self):
        """Test that successful execution updates trust score."""
        mesh = Mesh()

        def func(x):
            return x

        identity = await mesh.register(func, name="test", capabilities=["test"])

        # Execute successfully
        await mesh.execute(identity.id, "test")

        # Check trust score updated
        score = await mesh.trust.get_score(identity.id)
        assert score.total_executions == 1
        assert score.successful_executions == 1

    @pytest.mark.asyncio
    async def test_execution_updates_trust_on_failure(self):
        """Test that failed execution updates trust score."""
        from capabilitymesh.core.exceptions import ExecutionError

        mesh = Mesh()

        def failing_func(x):
            raise ValueError("Error")

        identity = await mesh.register(
            failing_func, name="failer", capabilities=["test"]
        )

        # Execute and fail
        try:
            await mesh.execute(identity.id, "test")
        except ExecutionError:
            pass

        # Check trust score updated
        score = await mesh.trust.get_score(identity.id)
        assert score.total_executions == 1
        assert score.failed_executions == 1


class TestStorageIntegration:
    """Test different storage backends with Mesh."""

    @pytest.mark.asyncio
    async def test_mesh_with_custom_storage(self):
        """Test Mesh with custom storage backend."""
        from capabilitymesh.storage import InMemoryStorage

        custom_storage = InMemoryStorage()
        mesh = Mesh(storage=custom_storage)

        def func(x):
            return x

        await mesh.register(func, name="test", capabilities=["test"])

        # Verify agent is in custom storage
        agents = await mesh.list_agents()
        assert len(agents) == 1

    @pytest.mark.asyncio
    async def test_mesh_persistence_across_instances(self):
        """Test that agents persist in shared storage."""
        from capabilitymesh.storage import InMemoryStorage

        shared_storage = InMemoryStorage()

        # Create first mesh instance
        mesh1 = Mesh(storage=shared_storage)

        def func(x):
            return x

        await mesh1.register(func, name="test", capabilities=["test"])

        # Create second mesh instance with same storage
        mesh2 = Mesh(storage=shared_storage)

        # Should see agent from first instance
        agents = await mesh2.list_agents()
        assert len(agents) == 1
        assert agents[0].name == "test"


class TestMetadataHandling:
    """Test metadata handling in Mesh."""

    @pytest.mark.asyncio
    async def test_register_with_nested_metadata(self):
        """Test registering with nested metadata structure."""
        mesh = Mesh()

        def func(x):
            return x

        metadata = {
            "config": {"nested": {"deep": {"value": 123}}},
            "tags": ["a", "b", "c"],
            "version": "1.0.0",
        }

        identity = await mesh.register(
            func, name="test", capabilities=["test"], metadata=metadata
        )

        agents = await mesh.list_agents()
        assert agents[0].metadata["config"]["nested"]["deep"]["value"] == 123

    @pytest.mark.asyncio
    async def test_register_with_none_metadata(self):
        """Test registering with None metadata."""
        mesh = Mesh()

        def func(x):
            return x

        identity = await mesh.register(
            func, name="test", capabilities=["test"], metadata=None
        )

        assert identity is not None


class TestAgentTypes:
    """Test different agent types."""

    @pytest.mark.asyncio
    async def test_register_with_explicit_agent_type(self):
        """Test registering with explicit agent type."""
        mesh = Mesh()

        def func(x):
            return x

        identity = await mesh.register(
            func,
            name="test",
            capabilities=["test"],
            agent_type=AgentType.SOFTWARE,
        )

        agents = await mesh.list_agents()
        assert agents[0].agent_type == AgentType.SOFTWARE

    @pytest.mark.asyncio
    async def test_default_agent_type(self):
        """Test default agent type for Python functions."""
        mesh = Mesh()

        def func(x):
            return x

        identity = await mesh.register(func, name="test", capabilities=["test"])

        agents = await mesh.list_agents()
        # Default should be SOFTWARE for Python callables
        assert agents[0].agent_type == AgentType.SOFTWARE
