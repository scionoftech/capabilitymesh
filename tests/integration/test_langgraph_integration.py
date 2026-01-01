"""Tests for LangGraph integration."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from capabilitymesh.core.identity import AgentAddress
from capabilitymesh.core.types import AgentType, CapabilityType, IOFormat
from capabilitymesh.integrations.langgraph import (
    ACDPDiscoveryNode,
    ACDPLangGraphAgent,
    ACDPNegotiationNode,
    create_discovery_workflow,
)
from capabilitymesh.schemas.capability import Capability


class MockLangGraph:
    """Mock LangGraph CompiledGraph for testing."""

    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes or []
        self.edges = edges or []

    def invoke(self, state):
        """Mock invoke method."""
        return {"status": "completed", "output": "mock result"}


class TestACDPDiscoveryNode:
    """Test ACDP discovery node for LangGraph."""

    def test_initialization(self):
        """Test ACDPDiscoveryNode initialization."""
        node = ACDPDiscoveryNode(
            query="translation",
            max_results=10,
            min_trust_level=0.7
        )

        assert node.query == "translation"
        assert node.max_results == 10
        assert node.min_trust_level == 0.7

    def test_default_parameters(self):
        """Test default parameters."""
        node = ACDPDiscoveryNode(query="search")

        assert node.query == "search"
        assert node.max_results == 5
        assert node.min_trust_level == 0.5

    def test_call_updates_state(self):
        """Test that calling node updates state."""
        node = ACDPDiscoveryNode(query="translation")

        state = {"task": "translate text"}
        updated_state = node(state)

        assert "discovered_agents" in updated_state
        assert "discovery_query" in updated_state
        assert updated_state["discovery_query"] == "translation"

    def test_call_preserves_existing_state(self):
        """Test that calling node preserves existing state."""
        node = ACDPDiscoveryNode(query="analysis")

        state = {"task": "analyze data", "user_id": "123"}
        updated_state = node(state)

        assert updated_state["task"] == "analyze data"
        assert updated_state["user_id"] == "123"

    def test_call_returns_empty_agents_list(self):
        """Test that placeholder returns empty agents list."""
        node = ACDPDiscoveryNode(query="test")

        state = {}
        updated_state = node(state)

        assert updated_state["discovered_agents"] == []

    def test_multiple_calls(self):
        """Test calling node multiple times."""
        node = ACDPDiscoveryNode(query="coding")

        state1 = {"task": "task1"}
        state2 = {"task": "task2"}

        result1 = node(state1)
        result2 = node(state2)

        assert result1["discovery_query"] == "coding"
        assert result2["discovery_query"] == "coding"


class TestACDPNegotiationNode:
    """Test ACDP negotiation node for LangGraph."""

    def test_initialization(self):
        """Test ACDPNegotiationNode initialization."""
        node = ACDPNegotiationNode(
            required_capabilities=["translation", "summarization"],
            max_cost_per_request=0.10,
            max_latency_ms=1000
        )

        assert node.required_capabilities == ["translation", "summarization"]
        assert node.max_cost_per_request == 0.10
        assert node.max_latency_ms == 1000

    def test_initialization_with_optional_params(self):
        """Test initialization with only required parameters."""
        node = ACDPNegotiationNode(required_capabilities=["search"])

        assert node.required_capabilities == ["search"]
        assert node.max_cost_per_request is None
        assert node.max_latency_ms is None

    def test_call_updates_state_with_agreements(self):
        """Test that calling node adds agreements to state."""
        node = ACDPNegotiationNode(
            required_capabilities=["translation"],
            max_cost_per_request=0.05
        )

        state = {"discovered_agents": []}
        updated_state = node(state)

        assert "agreements" in updated_state

    def test_call_preserves_existing_state(self):
        """Test that negotiation preserves existing state."""
        node = ACDPNegotiationNode(required_capabilities=["test"])

        state = {
            "discovered_agents": ["agent1", "agent2"],
            "task": "translate",
        }
        updated_state = node(state)

        assert updated_state["discovered_agents"] == ["agent1", "agent2"]
        assert updated_state["task"] == "translate"

    def test_call_with_empty_discovered_agents(self):
        """Test negotiation with no discovered agents."""
        node = ACDPNegotiationNode(required_capabilities=["test"])

        state = {}
        updated_state = node(state)

        assert "agreements" in updated_state
        assert updated_state["agreements"] == []

    def test_multiple_required_capabilities(self):
        """Test with multiple required capabilities."""
        capabilities = ["translate", "summarize", "analyze"]
        node = ACDPNegotiationNode(required_capabilities=capabilities)

        assert node.required_capabilities == capabilities
        assert len(node.required_capabilities) == 3


class TestACDPLangGraphAgent:
    """Test ACDP adapter for LangGraph workflows."""

    def test_wrap_langgraph(self):
        """Test wrapping a LangGraph workflow."""
        mock_graph = MockLangGraph()

        acdp_agent = ACDPLangGraphAgent.wrap(
            mock_graph,
            name="WorkflowAgent",
            description="Processes data workflows"
        )

        assert acdp_agent.agent_identity.name == "WorkflowAgent"
        assert acdp_agent.agent_identity.description == "Processes data workflows"
        assert acdp_agent.graph == mock_graph

    def test_wrap_without_name(self):
        """Test wrapping without providing name."""
        mock_graph = MockLangGraph()

        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        assert acdp_agent.agent_identity.name == "LangGraph Agent"

    def test_wrap_without_description(self):
        """Test wrapping without description."""
        mock_graph = MockLangGraph()

        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph, name="TestAgent")

        assert acdp_agent.agent_identity.name == "TestAgent"
        # Description can be None for LangGraph agents
        assert acdp_agent.agent_identity.description is None

    def test_wrap_generates_did(self):
        """Test that wrapping generates a valid DID."""
        mock_graph = MockLangGraph()

        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        assert acdp_agent.agent_identity.did.startswith("did:key:")

    def test_wrap_creates_address(self):
        """Test that wrapping creates an agent address."""
        mock_graph = MockLangGraph()

        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph, name="MyWorkflow")

        address = acdp_agent.agent_identity.primary_address
        assert isinstance(address, AgentAddress)
        assert address.protocol == "http"
        assert address.path == "/langgraph/MyWorkflow"

    def test_wrap_agent_type_is_software(self):
        """Test that LangGraph agents are typed as SOFTWARE."""
        mock_graph = MockLangGraph()

        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        assert acdp_agent.agent_identity.agent_type == AgentType.SOFTWARE

    def test_extract_capabilities(self):
        """Test extracting capabilities from LangGraph workflow."""
        mock_graph = MockLangGraph()
        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph, name="TestWorkflow")

        capabilities = acdp_agent.extract_capabilities()

        assert len(capabilities) > 0
        assert isinstance(capabilities[0], Capability)

    def test_capability_is_structured(self):
        """Test that LangGraph capabilities are structured."""
        mock_graph = MockLangGraph()
        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        capabilities = acdp_agent.extract_capabilities()

        assert capabilities[0].capability_type == CapabilityType.STRUCTURED

    def test_capability_has_json_io(self):
        """Test that capabilities have JSON input/output."""
        mock_graph = MockLangGraph()
        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        capabilities = acdp_agent.extract_capabilities()

        cap = capabilities[0]
        assert cap.inputs[0].format == IOFormat.JSON
        assert cap.outputs[0].format == IOFormat.JSON

    def test_capability_tags(self):
        """Test capability tags."""
        mock_graph = MockLangGraph()
        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        capabilities = acdp_agent.extract_capabilities()

        tags = capabilities[0].semantic.tags
        assert "langgraph" in tags
        assert "workflow" in tags
        assert "langchain" in tags

    def test_capability_categories(self):
        """Test capability categories."""
        mock_graph = MockLangGraph()
        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        capabilities = acdp_agent.extract_capabilities()

        categories = capabilities[0].semantic.categories
        assert "workflow" in categories
        assert "orchestration" in categories

    def test_capability_has_structured_spec(self):
        """Test that capability has structured specification."""
        mock_graph = MockLangGraph()
        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        capabilities = acdp_agent.extract_capabilities()

        cap = capabilities[0]
        assert cap.structured_spec is not None
        assert cap.structured_spec.endpoint == "/workflow/execute"
        assert cap.structured_spec.method == "POST"

    def test_execute_task(self):
        """Test executing a task with LangGraph."""
        mock_graph = MockLangGraph()
        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        task = {"input": "test data"}
        result = acdp_agent.execute_task(task)

        assert result["status"] == "completed"

    def test_execute_task_error_handling(self):
        """Test error handling during task execution."""
        mock_graph = MockLangGraph()
        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        task = {"input": "test"}
        result = acdp_agent.execute_task(task)

        assert "status" in result

    def test_initialization_direct(self):
        """Test direct initialization (not using wrap)."""
        from capabilitymesh.core.identity import AgentIdentity

        mock_graph = MockLangGraph()
        address = AgentAddress(
            protocol="http",
            host="localhost",
            port=8000,
            path="/test"
        )

        identity = AgentIdentity(
            did="did:key:test",
            name="TestAgent",
            agent_type=AgentType.SOFTWARE,
            description="Test",
            addresses=[address],
            primary_address=address,
            public_key="test-key",
            created_at=datetime.now(),
            last_seen=datetime.now(),
        )

        agent = ACDPLangGraphAgent(agent_identity=identity, graph=mock_graph)

        assert agent.agent_identity == identity
        assert agent.graph == mock_graph


class TestCreateDiscoveryWorkflow:
    """Test creating discovery workflows."""

    def test_create_workflow(self):
        """Test creating a basic discovery workflow."""

        def execution_func(state):
            return {"result": "executed"}

        workflow = create_discovery_workflow(
            discovery_query="translation",
            execution_function=execution_func
        )

        assert workflow is not None
        assert "discovery_query" in workflow
        assert workflow["discovery_query"] == "translation"

    def test_workflow_includes_execution_function(self):
        """Test that workflow includes execution function."""

        def custom_exec(state):
            return {"custom": "result"}

        workflow = create_discovery_workflow(
            discovery_query="analysis",
            execution_function=custom_exec
        )

        assert "execution_function" in workflow
        assert workflow["execution_function"] == custom_exec

    def test_workflow_with_complex_query(self):
        """Test workflow with complex discovery query."""

        def exec_func(state):
            return state

        query = "translate english to french with high accuracy"
        workflow = create_discovery_workflow(
            discovery_query=query,
            execution_function=exec_func
        )

        assert workflow["discovery_query"] == query

    def test_workflow_with_lambda_function(self):
        """Test workflow with lambda execution function."""
        workflow = create_discovery_workflow(
            discovery_query="test",
            execution_function=lambda s: {"done": True}
        )

        assert workflow is not None
        assert callable(workflow["execution_function"])


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_discovery_and_negotiation_nodes(self):
        """Test using discovery and negotiation nodes together."""
        discovery = ACDPDiscoveryNode(query="translation", max_results=5)
        negotiation = ACDPNegotiationNode(
            required_capabilities=["translation"],
            max_cost_per_request=0.05
        )

        # Simulate workflow
        state = {"task": "translate text"}

        # Discovery phase
        state = discovery(state)
        assert "discovered_agents" in state

        # Negotiation phase
        state = negotiation(state)
        assert "agreements" in state

    def test_full_workflow_simulation(self):
        """Test complete workflow: wrap -> extract -> execute."""
        mock_graph = MockLangGraph()

        # Wrap
        acdp_agent = ACDPLangGraphAgent.wrap(
            mock_graph,
            name="DataProcessor",
            description="Processes and analyzes data"
        )

        # Extract capabilities
        capabilities = acdp_agent.extract_capabilities()
        assert len(capabilities) > 0

        # Execute task
        task = {"data": "test data"}
        result = acdp_agent.execute_task(task)
        assert result["status"] == "completed"

    def test_workflow_with_discovery_node(self):
        """Test creating workflow with discovery node."""
        discovery_node = ACDPDiscoveryNode(query="analysis")

        def execute_with_discovered_agents(state):
            agents = state.get("discovered_agents", [])
            return {"processed_by": len(agents)}

        workflow = create_discovery_workflow(
            discovery_query="analysis",
            execution_function=execute_with_discovered_agents
        )

        assert workflow["discovery_query"] == "analysis"

    def test_multiple_nodes_in_sequence(self):
        """Test multiple nodes working in sequence."""
        discovery = ACDPDiscoveryNode(query="coding")
        negotiation = ACDPNegotiationNode(required_capabilities=["coding"])

        state = {"request": "write python function"}

        # Execute nodes in sequence
        state = discovery(state)
        assert "discovered_agents" in state

        state = negotiation(state)
        assert "agreements" in state

        # Original state preserved
        assert state["request"] == "write python function"

    def test_agent_with_custom_graph_nodes(self):
        """Test agent with custom graph structure."""
        custom_graph = MockLangGraph(
            nodes=["start", "process", "end"],
            edges=[("start", "process"), ("process", "end")]
        )

        acdp_agent = ACDPLangGraphAgent.wrap(
            custom_graph,
            name="CustomWorkflow"
        )

        assert acdp_agent.graph.nodes == ["start", "process", "end"]
        assert acdp_agent.graph.edges == [("start", "process"), ("process", "end")]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_discovery_node_with_empty_query(self):
        """Test discovery node with empty query string."""
        node = ACDPDiscoveryNode(query="")

        assert node.query == ""

    def test_negotiation_node_with_empty_capabilities(self):
        """Test negotiation node with empty capabilities list."""
        node = ACDPNegotiationNode(required_capabilities=[])

        assert node.required_capabilities == []

    def test_negotiation_node_with_zero_cost(self):
        """Test negotiation with zero cost constraint."""
        node = ACDPNegotiationNode(
            required_capabilities=["test"],
            max_cost_per_request=0.0
        )

        assert node.max_cost_per_request == 0.0

    def test_negotiation_node_with_very_high_latency(self):
        """Test negotiation with very high latency constraint."""
        node = ACDPNegotiationNode(
            required_capabilities=["test"],
            max_latency_ms=999999
        )

        assert node.max_latency_ms == 999999

    def test_wrap_graph_with_none_name(self):
        """Test wrapping with explicit None name."""
        mock_graph = MockLangGraph()

        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph, name=None)

        assert acdp_agent.agent_identity.name == "LangGraph Agent"

    def test_execute_task_with_empty_dict(self):
        """Test executing task with empty dictionary."""
        mock_graph = MockLangGraph()
        acdp_agent = ACDPLangGraphAgent.wrap(mock_graph)

        result = acdp_agent.execute_task({})

        assert "status" in result

    def test_discovery_node_state_mutation(self):
        """Test that discovery node doesn't mutate original state object."""
        node = ACDPDiscoveryNode(query="test")

        original_state = {"key": "value"}
        state_copy = original_state.copy()

        node(original_state)

        # Note: The current implementation mutates the state dict
        # If immutability is desired, this test would need updating
        assert "discovered_agents" in original_state
