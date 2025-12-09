"""Integration tests for CrewAI adapter."""

import pytest
from datetime import datetime

from capabilitymesh.integrations.crewai import ACDPCrewAIAgent, DynamicCrew
from capabilitymesh.core.types import AgentType, CapabilityType


class MockCrewAIAgent:
    """Mock CrewAI agent for testing."""

    def __init__(self, role, goal, backstory, tools=None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []


class TestACDPCrewAIAgent:
    """Tests for ACDPCrewAIAgent integration."""

    @pytest.fixture
    def mock_crew_agent(self):
        """Create a mock CrewAI agent."""
        return MockCrewAIAgent(
            role="Data Analyst",
            goal="Analyze data and provide insights",
            backstory="Expert data analyst with 10 years of experience",
            tools=["pandas", "numpy"]
        )

    def test_wrap_crew_agent(self, mock_crew_agent):
        """Test wrapping a CrewAI agent."""
        acdp_agent = ACDPCrewAIAgent.wrap(mock_crew_agent)

        assert acdp_agent is not None
        assert acdp_agent.agent_identity.name == "Data Analyst"
        assert acdp_agent.agent_identity.agent_type == AgentType.LLM
        assert acdp_agent.crew_agent == mock_crew_agent

    def test_extract_capabilities(self, mock_crew_agent):
        """Test capability extraction from CrewAI agent."""
        acdp_agent = ACDPCrewAIAgent.wrap(mock_crew_agent)
        capabilities = acdp_agent.extract_capabilities()

        assert len(capabilities) > 0
        cap = capabilities[0]
        assert cap.name == "data-analyst"
        assert cap.capability_type == CapabilityType.UNSTRUCTURED
        assert cap.agent_type == AgentType.LLM
        assert "pandas" in cap.semantic.tags or "numpy" in cap.semantic.tags

    def test_register_auto_capabilities(self, mock_crew_agent):
        """Test auto-registration of capabilities."""
        acdp_agent = ACDPCrewAIAgent.wrap(mock_crew_agent)
        registered = acdp_agent.register_auto_capabilities()

        assert len(registered) > 0
        assert len(acdp_agent.capabilities) == len(registered)
        assert acdp_agent.capabilities[0] == registered[0]

    def test_execute_task(self, mock_crew_agent):
        """Test task execution."""
        acdp_agent = ACDPCrewAIAgent.wrap(mock_crew_agent)
        task = {"description": "Analyze sales data"}
        result = acdp_agent.execute_task(task)

        assert result is not None
        assert "status" in result

    def test_to_dict(self, mock_crew_agent):
        """Test converting adapter to dictionary."""
        acdp_agent = ACDPCrewAIAgent.wrap(mock_crew_agent)
        acdp_agent.register_auto_capabilities()
        agent_dict = acdp_agent.to_dict()

        assert "agent_identity" in agent_dict
        assert "framework" in agent_dict
        assert agent_dict["framework"] == "ACDPCrewAIAgent"
        assert "capabilities" in agent_dict
        assert len(agent_dict["capabilities"]) > 0


class TestDynamicCrew:
    """Tests for DynamicCrew manager."""

    def test_create_dynamic_crew(self):
        """Test creating a dynamic crew manager."""
        crew_manager = DynamicCrew()
        assert crew_manager is not None
        assert len(crew_manager.discovered_agents) == 0

    def test_add_agent(self):
        """Test adding an agent to the crew."""
        crew_manager = DynamicCrew()
        mock_agent = MockCrewAIAgent("Researcher", "Research topics", "Expert researcher")
        acdp_agent = ACDPCrewAIAgent.wrap(mock_agent)

        crew_manager.add_agent(acdp_agent)
        assert len(crew_manager.discovered_agents) == 1
        assert crew_manager.discovered_agents[0] == acdp_agent

    def test_form_crew(self):
        """Test forming a crew from agents."""
        crew_manager = DynamicCrew()
        agent1 = ACDPCrewAIAgent.wrap(MockCrewAIAgent("Agent1", "Goal1", "Story1"))
        agent2 = ACDPCrewAIAgent.wrap(MockCrewAIAgent("Agent2", "Goal2", "Story2"))

        agent1.register_auto_capabilities()
        agent2.register_auto_capabilities()

        crew_config = crew_manager.form_crew([agent1, agent2])

        assert "agents" in crew_config
        assert len(crew_config["agents"]) == 2
        assert "capabilities" in crew_config
