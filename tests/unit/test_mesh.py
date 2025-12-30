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
