"""Tests for AutoGen integration."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from capabilitymesh.core.identity import AgentAddress
from capabilitymesh.core.types import AgentType, CapabilityType, IOFormat
from capabilitymesh.integrations.autogen import (
    ACDPAutoGenAgent,
    CapabilityBasedRouter,
    DynamicGroupChat,
)
from capabilitymesh.schemas.capability import Capability


class MockAutoGenAgent:
    """Mock AutoGen ConversableAgent for testing."""

    def __init__(
        self,
        name: str = "TestAgent",
        system_message: str = "You are a helpful assistant",
        llm_config: dict = None,
        function_map: dict = None
    ):
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config or {"model": "gpt-4"}
        self._function_map = function_map or {}


class TestACDPAutoGenAgentWrapping:
    """Test wrapping AutoGen agents with ACDP."""

    def test_wrap_autogen_agent(self):
        """Test wrapping an AutoGen agent."""
        mock_agent = MockAutoGenAgent(name="DataAnalyst")

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        assert acdp_agent.agent_identity.name == "DataAnalyst"
        assert acdp_agent.agent_identity.agent_type == AgentType.LLM
        assert acdp_agent.autogen_agent == mock_agent

    def test_wrap_with_custom_name(self):
        """Test wrapping with custom name override."""
        mock_agent = MockAutoGenAgent(name="OriginalName")

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent, name="CustomName")

        assert acdp_agent.agent_identity.name == "CustomName"

    def test_wrap_agent_without_name(self):
        """Test wrapping agent without name attribute."""
        mock_agent = MagicMock()
        del mock_agent.name  # Remove name attribute

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        assert acdp_agent.agent_identity.name == "AutoGen Agent"

    def test_wrap_generates_did(self):
        """Test that wrapping generates a valid DID."""
        mock_agent = MockAutoGenAgent()

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        assert acdp_agent.agent_identity.did.startswith("did:key:")

    def test_wrap_creates_address(self):
        """Test that wrapping creates an agent address."""
        mock_agent = MockAutoGenAgent(name="TestAgent")

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        address = acdp_agent.agent_identity.primary_address
        assert isinstance(address, AgentAddress)
        assert address.protocol == "http"
        assert address.path == "/autogen/testagent"

    def test_wrap_uses_system_message_as_description(self):
        """Test that system message is used as description."""
        system_message = "You are a data analysis expert"
        mock_agent = MockAutoGenAgent(system_message=system_message)

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        assert system_message in acdp_agent.agent_identity.description

    def test_wrap_truncates_long_system_message(self):
        """Test that long system messages are truncated."""
        long_message = "A" * 300
        mock_agent = MockAutoGenAgent(system_message=long_message)

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        assert len(acdp_agent.agent_identity.description) == 200

    def test_wrap_handles_empty_system_message(self):
        """Test wrapping agent with empty system message."""
        mock_agent = MockAutoGenAgent(system_message="")

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        assert "AutoGen agent" in acdp_agent.agent_identity.description


class TestCapabilityExtraction:
    """Test capability extraction from AutoGen agents."""

    def test_extract_capabilities_basic(self):
        """Test basic capability extraction."""
        mock_agent = MockAutoGenAgent(
            name="Translator",
            system_message="I translate text between languages"
        )

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        assert len(capabilities) > 0
        assert isinstance(capabilities[0], Capability)

    def test_capability_name_derived_from_agent_name(self):
        """Test that capability name is derived from agent name."""
        mock_agent = MockAutoGenAgent(name="Data Analyst")

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        assert capabilities[0].name == "data-analyst"

    def test_capability_description_from_system_message(self):
        """Test that capability description comes from system message."""
        system_message = "Expert in data analysis and visualization"
        mock_agent = MockAutoGenAgent(system_message=system_message)

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        assert capabilities[0].description == system_message

    def test_capability_type_is_unstructured(self):
        """Test that extracted capabilities are unstructured type."""
        mock_agent = MockAutoGenAgent()

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        assert capabilities[0].capability_type == CapabilityType.UNSTRUCTURED

    def test_capability_has_text_io_formats(self):
        """Test that capabilities have text input/output formats."""
        mock_agent = MockAutoGenAgent(name="TestAgent")

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        cap = capabilities[0]
        assert cap.inputs[0].format == IOFormat.TEXT
        assert cap.outputs[0].format == IOFormat.TEXT

    def test_capability_includes_llm_config(self):
        """Test that capability includes LLM configuration."""
        llm_config = {"model": "gpt-4", "temperature": 0.7}
        mock_agent = MockAutoGenAgent(llm_config=llm_config)

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        assert capabilities[0].unstructured_spec.model_info["config"] == llm_config

    def test_capability_tags_from_name(self):
        """Test that tags are extracted from agent name."""
        mock_agent = MockAutoGenAgent(name="DataAnalyst")

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        tags = capabilities[0].semantic.tags
        assert "dataanalyst" in tags

    def test_capability_tags_from_system_message(self):
        """Test that tags are extracted from system message."""
        mock_agent = MockAutoGenAgent(
            name="Agent",
            system_message="You are an expert analyst and developer"
        )

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        tags = capabilities[0].semantic.tags
        assert "analyst" in tags
        assert "developer" in tags

    def test_capability_categories(self):
        """Test that capability has appropriate categories."""
        mock_agent = MockAutoGenAgent()

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        categories = capabilities[0].semantic.categories
        assert "autogen" in categories
        assert "agent" in categories
        assert "llm" in categories

    def test_capability_includes_function_map(self):
        """Test that function map is included in tags."""
        function_map = {"search_web": MagicMock(), "analyze_data": MagicMock()}
        mock_agent = MockAutoGenAgent(function_map=function_map)

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        tags = capabilities[0].semantic.tags
        assert "search_web" in tags
        assert "analyze_data" in tags

    def test_extract_tags_deduplication(self):
        """Test that duplicate tags are removed."""
        mock_agent = MockAutoGenAgent(
            name="analyst",
            system_message="You are a data analyst expert"
        )

        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        capabilities = acdp_agent.extract_capabilities()

        tags = capabilities[0].semantic.tags
        # "analyst" appears in name and message, should only appear once
        assert tags.count("analyst") == 1


class TestTaskExecution:
    """Test task execution with AutoGen agents."""

    def test_execute_task_with_message(self):
        """Test executing a task with message field."""
        mock_agent = MockAutoGenAgent()
        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        task = {"message": "Analyze this data"}
        result = acdp_agent.execute_task(task)

        assert result["status"] == "completed"
        assert "response" in result

    def test_execute_task_with_description(self):
        """Test executing a task with description field (fallback)."""
        mock_agent = MockAutoGenAgent()
        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        task = {"description": "Translate this text"}
        result = acdp_agent.execute_task(task)

        assert result["status"] == "completed"

    def test_execute_task_error_handling(self):
        """Test error handling during task execution."""
        mock_agent = MockAutoGenAgent()
        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        # This will use the placeholder implementation which doesn't raise
        task = {"message": "Test"}
        result = acdp_agent.execute_task(task)

        assert "status" in result


class TestDynamicGroupChat:
    """Test dynamic group chat functionality."""

    def test_initialization(self):
        """Test DynamicGroupChat initialization."""
        chat = DynamicGroupChat()

        assert chat.discovered_agents == []

    def test_add_agent(self):
        """Test adding an agent to group chat."""
        chat = DynamicGroupChat()
        mock_agent = MockAutoGenAgent()
        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        chat.add_agent(acdp_agent)

        assert len(chat.discovered_agents) == 1
        assert chat.discovered_agents[0] == acdp_agent

    def test_add_multiple_agents(self):
        """Test adding multiple agents."""
        chat = DynamicGroupChat()

        agents = [
            ACDPAutoGenAgent.wrap(MockAutoGenAgent(name=f"Agent{i}"))
            for i in range(3)
        ]

        for agent in agents:
            chat.add_agent(agent)

        assert len(chat.discovered_agents) == 3

    def test_create_group_chat(self):
        """Test creating a group chat configuration."""
        chat = DynamicGroupChat()
        mock_agent = MockAutoGenAgent()
        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        chat.add_agent(acdp_agent)

        config = chat.create_group_chat()

        assert "agents" in config
        assert "capabilities" in config
        assert config["agents"][0] == mock_agent

    def test_discover_and_add_placeholder(self):
        """Test discover_and_add returns None (placeholder)."""
        chat = DynamicGroupChat()

        result = chat.discover_and_add("translation")

        assert result is None


class TestCapabilityBasedRouter:
    """Test capability-based routing functionality."""

    def test_initialization(self):
        """Test CapabilityBasedRouter initialization."""
        router = CapabilityBasedRouter()

        assert router.agents == []

    def test_register_agent(self):
        """Test registering an agent with router."""
        router = CapabilityBasedRouter()
        mock_agent = MockAutoGenAgent()
        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        router.register_agent(acdp_agent)

        assert len(router.agents) == 1
        assert router.agents[0] == acdp_agent

    def test_register_multiple_agents(self):
        """Test registering multiple agents."""
        router = CapabilityBasedRouter()

        agents = [
            ACDPAutoGenAgent.wrap(MockAutoGenAgent(name=f"Agent{i}"))
            for i in range(3)
        ]

        for agent in agents:
            router.register_agent(agent)

        assert len(router.agents) == 3

    def test_route_with_no_agents(self):
        """Test routing when no agents are registered."""
        router = CapabilityBasedRouter()

        result = router.route("Translate this text")

        assert result is None

    def test_route_with_agents(self):
        """Test routing with registered agents."""
        router = CapabilityBasedRouter()
        mock_agent = MockAutoGenAgent()
        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)
        router.register_agent(acdp_agent)

        result = router.route("Analyze this data")

        assert result is not None
        assert result == acdp_agent

    def test_route_with_score(self):
        """Test routing with confidence scores."""
        router = CapabilityBasedRouter()

        agents = [
            ACDPAutoGenAgent.wrap(MockAutoGenAgent(name=f"Agent{i}"))
            for i in range(3)
        ]

        for agent in agents:
            router.register_agent(agent)

        results = router.route_with_score("Test message")

        assert len(results) == 3
        # Each result is (agent, score) tuple
        for agent, score in results:
            assert isinstance(agent, ACDPAutoGenAgent)
            assert isinstance(score, float)

    def test_route_with_score_sorted(self):
        """Test that route_with_score returns sorted results."""
        router = CapabilityBasedRouter()

        agents = [
            ACDPAutoGenAgent.wrap(MockAutoGenAgent(name=f"Agent{i}"))
            for i in range(3)
        ]

        for agent in agents:
            router.register_agent(agent)

        results = router.route_with_score("Test message")

        # Verify results are sorted by score (descending)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_route_with_score_empty_agents(self):
        """Test route_with_score with no agents."""
        router = CapabilityBasedRouter()

        results = router.route_with_score("Test message")

        assert results == []


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test complete workflow: wrap -> extract -> execute."""
        mock_agent = MockAutoGenAgent(
            name="Translator",
            system_message="I translate text between languages",
            llm_config={"model": "gpt-4"}
        )

        # Wrap
        acdp_agent = ACDPAutoGenAgent.wrap(mock_agent)

        # Extract capabilities
        capabilities = acdp_agent.extract_capabilities()
        assert len(capabilities) > 0

        # Execute task
        task = {"message": "Translate 'Hello' to French"}
        result = acdp_agent.execute_task(task)
        assert result["status"] == "completed"

    def test_multiple_agents_in_router(self):
        """Test routing between multiple different agents."""
        translator = ACDPAutoGenAgent.wrap(
            MockAutoGenAgent(name="Translator", system_message="I translate text")
        )
        analyst = ACDPAutoGenAgent.wrap(
            MockAutoGenAgent(name="Analyst", system_message="I analyze data")
        )
        coder = ACDPAutoGenAgent.wrap(
            MockAutoGenAgent(name="Coder", system_message="I write code")
        )

        router = CapabilityBasedRouter()
        router.register_agent(translator)
        router.register_agent(analyst)
        router.register_agent(coder)

        # Route message
        result = router.route("Write Python code")

        assert result is not None
        assert result in [translator, analyst, coder]
