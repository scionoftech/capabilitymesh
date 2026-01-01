"""Tests for A2A client."""

import pytest

from capabilitymesh.core.identity import AgentAddress
from capabilitymesh.integrations.a2a.client import A2AClient, A2ATaskBuilder


class TestA2AClientInitialization:
    """Test A2A client initialization."""

    def test_initialization_with_url(self):
        """Test initializing client with agent URL."""
        client = A2AClient(agent_url="http://localhost:8000/agent")

        assert client.agent_url == "http://localhost:8000/agent"
        assert client.timeout == 300
        assert client.http_client is None

    def test_initialization_with_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = A2AClient(
            agent_url="http://example.com/agent",
            timeout=600
        )

        assert client.timeout == 600

    def test_initialization_with_http_client(self):
        """Test initialization with custom HTTP client."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        client = A2AClient(
            agent_url="http://example.com/agent",
            http_client=mock_client
        )

        assert client.http_client == mock_client

    def test_from_agent_address(self):
        """Test creating client from AgentAddress."""
        address = AgentAddress(
            protocol="https",
            host="example.com",
            port=443,
            path="/api/agent"
        )

        client = A2AClient.from_agent_address(address)

        assert client.agent_url == "https://example.com:443/api/agent"

    def test_from_agent_address_with_additional_kwargs(self):
        """Test from_agent_address with additional parameters."""
        address = AgentAddress(
            protocol="http",
            host="localhost",
            port=8000,
            path="/agent"
        )

        client = A2AClient.from_agent_address(address, timeout=120)

        assert client.timeout == 120

    def test_from_acdp_agreement(self):
        """Test creating client from ACDP agreement."""
        agreement = {
            "provider_address": "http://agent.example.com/api",
            "terms": {"cost": 0.01},
        }

        client = A2AClient.from_acdp_agreement(agreement)

        assert client.agent_url == "http://agent.example.com/api"

    def test_from_acdp_agreement_missing_address(self):
        """Test creating client from agreement without provider address."""
        agreement = {"terms": {"cost": 0.01}}

        client = A2AClient.from_acdp_agreement(agreement)

        assert client.agent_url == ""

    def test_from_acdp_agreement_empty(self):
        """Test creating client from empty agreement."""
        client = A2AClient.from_acdp_agreement({})

        assert client.agent_url == ""


class TestTaskCreation:
    """Test A2A task creation."""

    def test_create_task_basic(self):
        """Test creating a basic task."""
        client = A2AClient(agent_url="http://localhost:8000")

        task = client.create_task(instructions="Translate this text")

        assert task["instructions"] == "Translate this text"
        assert task["status"] == "pending"
        assert "taskId" in task
        assert isinstance(task["metadata"], dict)

    def test_create_task_with_task_id(self):
        """Test creating task with custom task ID."""
        client = A2AClient(agent_url="http://localhost:8000")

        task = client.create_task(
            instructions="Analyze data",
            task_id="custom-task-123"
        )

        assert task["taskId"] == "custom-task-123"

    def test_create_task_with_metadata(self):
        """Test creating task with metadata."""
        client = A2AClient(agent_url="http://localhost:8000")

        metadata = {"priority": "high", "user_id": "user-456"}
        task = client.create_task(
            instructions="Process request",
            metadata=metadata
        )

        assert task["metadata"] == metadata
        assert task["metadata"]["priority"] == "high"

    def test_create_task_generates_unique_ids(self):
        """Test that task IDs are unique."""
        client = A2AClient(agent_url="http://localhost:8000")

        task1 = client.create_task(instructions="Task 1")
        task2 = client.create_task(instructions="Task 2")

        assert task1["taskId"] != task2["taskId"]

    def test_create_task_id_format(self):
        """Test that generated task IDs have correct format."""
        client = A2AClient(agent_url="http://localhost:8000")

        task = client.create_task(instructions="Test")

        assert task["taskId"].startswith("task-")
        # Format is "task-" + 12 hex chars
        assert len(task["taskId"]) == 17  # "task-" (5) + 12 hex chars

    def test_create_task_empty_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        client = A2AClient(agent_url="http://localhost:8000")

        task = client.create_task(instructions="Test")

        assert task["metadata"] == {}

    def test_create_task_with_empty_instructions(self):
        """Test creating task with empty instructions."""
        client = A2AClient(agent_url="http://localhost:8000")

        task = client.create_task(instructions="")

        assert task["instructions"] == ""
        assert "taskId" in task


class TestTaskExecution:
    """Test A2A task execution methods."""

    @pytest.mark.asyncio
    async def test_execute_task_not_implemented(self):
        """Test that execute_task raises NotImplementedError."""
        client = A2AClient(agent_url="http://localhost:8000")
        task = client.create_task(instructions="Test task")

        with pytest.raises(NotImplementedError) as exc_info:
            await client.execute_task(task)

        assert "httpx dependency" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_task_status_not_implemented(self):
        """Test that get_task_status raises NotImplementedError."""
        client = A2AClient(agent_url="http://localhost:8000")

        with pytest.raises(NotImplementedError) as exc_info:
            await client.get_task_status("task-123")

        assert "httpx dependency" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_wait_for_completion_not_implemented(self):
        """Test that wait_for_completion raises NotImplementedError."""
        client = A2AClient(agent_url="http://localhost:8000")

        with pytest.raises(NotImplementedError) as exc_info:
            await client.wait_for_completion("task-123")

        assert "httpx dependency" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_wait_for_completion_with_custom_interval(self):
        """Test wait_for_completion with custom poll interval."""
        client = A2AClient(agent_url="http://localhost:8000")

        with pytest.raises(NotImplementedError):
            await client.wait_for_completion("task-123", poll_interval=5)

    def test_get_agent_card_not_implemented(self):
        """Test that get_agent_card raises NotImplementedError."""
        client = A2AClient(agent_url="http://localhost:8000")

        with pytest.raises(NotImplementedError) as exc_info:
            client.get_agent_card()

        assert "httpx dependency" in str(exc_info.value)


class TestA2ATaskBuilder:
    """Test A2A task builder."""

    def test_initialization(self):
        """Test task builder initialization."""
        builder = A2ATaskBuilder()

        assert "taskId" in builder.task
        assert builder.task["status"] == "pending"
        assert builder.task["metadata"] == {}

    def test_with_instructions(self):
        """Test setting instructions."""
        builder = A2ATaskBuilder()
        result = builder.with_instructions("Translate text")

        assert result is builder  # Check method chaining
        assert builder.task["instructions"] == "Translate text"

    def test_with_task_id(self):
        """Test setting custom task ID."""
        builder = A2ATaskBuilder()
        result = builder.with_task_id("custom-id-123")

        assert result is builder
        assert builder.task["taskId"] == "custom-id-123"

    def test_with_metadata(self):
        """Test adding metadata."""
        builder = A2ATaskBuilder()
        result = builder.with_metadata("priority", "high")

        assert result is builder
        assert builder.task["metadata"]["priority"] == "high"

    def test_with_multiple_metadata(self):
        """Test adding multiple metadata entries."""
        builder = A2ATaskBuilder()
        builder.with_metadata("priority", "high")
        builder.with_metadata("user_id", "user-456")

        assert builder.task["metadata"]["priority"] == "high"
        assert builder.task["metadata"]["user_id"] == "user-456"

    def test_with_context(self):
        """Test adding context."""
        builder = A2ATaskBuilder()
        context = {"session_id": "session-789", "language": "en"}
        result = builder.with_context(context)

        assert result is builder
        assert builder.task["context"] == context

    def test_build_success(self):
        """Test building a valid task."""
        builder = A2ATaskBuilder()
        task = builder.with_instructions("Process data").build()

        assert task["instructions"] == "Process data"
        assert "taskId" in task
        assert task["status"] == "pending"

    def test_build_without_instructions_raises_error(self):
        """Test that building without instructions raises ValueError."""
        builder = A2ATaskBuilder()

        with pytest.raises(ValueError) as exc_info:
            builder.build()

        assert "instructions" in str(exc_info.value)

    def test_method_chaining(self):
        """Test fluent interface with method chaining."""
        task = (A2ATaskBuilder()
                .with_instructions("Translate text")
                .with_task_id("task-abc")
                .with_metadata("priority", "high")
                .with_metadata("lang", "en")
                .with_context({"user": "john"})
                .build())

        assert task["instructions"] == "Translate text"
        assert task["taskId"] == "task-abc"
        assert task["metadata"]["priority"] == "high"
        assert task["metadata"]["lang"] == "en"
        assert task["context"]["user"] == "john"

    def test_generated_task_id_format(self):
        """Test that generated task ID has correct format."""
        builder = A2ATaskBuilder()

        task_id = builder.task["taskId"]
        assert task_id.startswith("task-")
        assert len(task_id) == 17

    def test_builder_creates_unique_ids(self):
        """Test that each builder creates unique task IDs."""
        builder1 = A2ATaskBuilder()
        builder2 = A2ATaskBuilder()

        assert builder1.task["taskId"] != builder2.task["taskId"]

    def test_overwrite_metadata(self):
        """Test that metadata can be overwritten."""
        builder = A2ATaskBuilder()
        builder.with_metadata("key", "value1")
        builder.with_metadata("key", "value2")

        assert builder.task["metadata"]["key"] == "value2"

    def test_with_empty_context(self):
        """Test adding empty context."""
        builder = A2ATaskBuilder()
        builder.with_context({})

        assert builder.task["context"] == {}

    def test_with_empty_instructions(self):
        """Test with empty instructions still allows build."""
        builder = A2ATaskBuilder()
        builder.with_instructions("")

        task = builder.build()
        assert task["instructions"] == ""

    def test_multiple_builds(self):
        """Test that build can be called multiple times."""
        builder = A2ATaskBuilder()
        builder.with_instructions("Test")

        task1 = builder.build()
        task2 = builder.build()

        # Both should return the same task dict
        assert task1 == task2

    def test_modify_after_build(self):
        """Test modifying builder after build."""
        builder = A2ATaskBuilder()
        builder.with_instructions("Original")

        task1 = builder.build()
        builder.with_instructions("Modified")
        task2 = builder.build()

        assert task1["instructions"] == "Modified"  # Shared reference
        assert task2["instructions"] == "Modified"


class TestIntegration:
    """Integration tests for A2A client and builder."""

    def test_client_with_builder_task(self):
        """Test using builder-created task with client."""
        client = A2AClient(agent_url="http://localhost:8000")

        task = (A2ATaskBuilder()
                .with_instructions("Translate text")
                .with_metadata("lang", "fr")
                .build())

        # Verify task structure is compatible
        assert "taskId" in task
        assert "instructions" in task
        assert task["status"] == "pending"

    def test_client_and_builder_task_ids_differ(self):
        """Test that client and builder generate different task IDs."""
        client = A2AClient(agent_url="http://localhost:8000")
        builder = A2ATaskBuilder()

        client_task = client.create_task(instructions="Test 1")
        builder_task = builder.with_instructions("Test 2").build()

        assert client_task["taskId"] != builder_task["taskId"]

    def test_client_from_address_and_create_task(self):
        """Test full workflow: create client from address and create task."""
        address = AgentAddress(
            protocol="https",
            host="agent.example.com",
            port=443,
            path="/api/v1/agent"
        )

        client = A2AClient.from_agent_address(address, timeout=60)
        task = client.create_task(
            instructions="Process request",
            metadata={"user": "alice"}
        )

        assert client.agent_url == "https://agent.example.com:443/api/v1/agent"
        assert client.timeout == 60
        assert task["instructions"] == "Process request"
        assert task["metadata"]["user"] == "alice"

    def test_builder_comprehensive_task(self):
        """Test creating comprehensive task with all features."""
        task = (A2ATaskBuilder()
                .with_task_id("comprehensive-task-001")
                .with_instructions("Analyze dataset and generate report")
                .with_metadata("priority", "high")
                .with_metadata("deadline", "2024-12-31")
                .with_metadata("requester", "data-team")
                .with_context({
                    "dataset_id": "ds-123",
                    "format": "pdf",
                    "language": "en"
                })
                .build())

        # Verify all fields
        assert task["taskId"] == "comprehensive-task-001"
        assert task["instructions"] == "Analyze dataset and generate report"
        assert task["metadata"]["priority"] == "high"
        assert task["metadata"]["deadline"] == "2024-12-31"
        assert task["metadata"]["requester"] == "data-team"
        assert task["context"]["dataset_id"] == "ds-123"
        assert task["context"]["format"] == "pdf"
        assert task["context"]["language"] == "en"
        assert task["status"] == "pending"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_client_with_empty_url(self):
        """Test creating client with empty URL."""
        client = A2AClient(agent_url="")

        assert client.agent_url == ""

    def test_client_with_very_long_url(self):
        """Test client with extremely long URL."""
        long_url = "http://example.com/" + "a" * 10000
        client = A2AClient(agent_url=long_url)

        assert client.agent_url == long_url

    def test_task_with_special_characters_in_instructions(self):
        """Test task with special characters."""
        client = A2AClient(agent_url="http://localhost:8000")

        instructions = "Translate: Hello! @#$%^&*() 你好 émoji 🚀"
        task = client.create_task(instructions=instructions)

        assert task["instructions"] == instructions

    def test_builder_with_complex_metadata_values(self):
        """Test builder with complex nested metadata."""
        builder = A2ATaskBuilder()
        builder.with_metadata("config", {"nested": {"deep": {"value": 123}}})
        builder.with_metadata("list", [1, 2, 3, 4, 5])

        task = builder.with_instructions("Test").build()

        assert task["metadata"]["config"]["nested"]["deep"]["value"] == 123
        assert task["metadata"]["list"] == [1, 2, 3, 4, 5]

    def test_zero_timeout(self):
        """Test client with zero timeout."""
        client = A2AClient(agent_url="http://localhost:8000", timeout=0)

        assert client.timeout == 0

    def test_negative_timeout(self):
        """Test client with negative timeout."""
        client = A2AClient(agent_url="http://localhost:8000", timeout=-1)

        assert client.timeout == -1

    def test_very_large_timeout(self):
        """Test client with very large timeout."""
        client = A2AClient(agent_url="http://localhost:8000", timeout=999999)

        assert client.timeout == 999999
