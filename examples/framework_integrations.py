"""
Comprehensive example demonstrating CapabilityMesh integration with multiple frameworks.

This example shows how CapabilityMesh acts as a universal discovery and negotiation layer
that works with A2A, CrewAI, AutoGen, and LangGraph.

Note: This is a demonstration of the API. Full functionality requires installing
the respective framework packages:
    pip install capabilitymesh[frameworks]
"""

from datetime import datetime
from uuid import uuid4

# CapabilityMesh core imports
from capabilitymesh import AgentIdentity, AgentAddress, Capability, CapabilityVersion
from capabilitymesh.core.types import AgentType, CapabilityType, IOFormat
from capabilitymesh.schemas.capability import CapabilityInputOutput, UnstructuredCapability

# Framework integration imports
from capabilitymesh.integrations.a2a import AgentCardConverter, A2AAdapter
from capabilitymesh.integrations.crewai import ACDPCrewAIAgent, DynamicCrew
from capabilitymesh.integrations.autogen import ACDPAutoGenAgent, DynamicGroupChat
from capabilitymesh.integrations.langgraph import ACDPDiscoveryNode, ACDPLangGraphAgent


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def example_a2a_integration():
    """Example 1: A2A Protocol Integration."""
    print_section("Example 1: A2A Protocol Integration")

    # Create a CapabilityMesh capability
    capability = Capability(
        id="cap-translate-en-fr",
        name="translate-en-fr",
        description="Translate English to French",
        version=CapabilityVersion(major=1, minor=0, patch=0),
        capability_type=CapabilityType.UNSTRUCTURED,
        agent_type=AgentType.LLM,
        inputs=[
            CapabilityInputOutput(
                format=IOFormat.TEXT,
                description="English text to translate",
            )
        ],
        outputs=[
            CapabilityInputOutput(
                format=IOFormat.TEXT,
                description="French translation",
            )
        ],
        unstructured_spec=UnstructuredCapability(
            prompt_template="Translate to French: {text}",
        ),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Create agent identity
    address = AgentAddress(protocol="http", host="localhost", port=8000)
    public_key = f"-----BEGIN PUBLIC KEY-----\n{uuid4().hex}\n-----END PUBLIC KEY-----"
    did = AgentIdentity.generate_did(public_key)

    agent_identity = AgentIdentity(
        did=did,
        name="TranslationAgent",
        agent_type=AgentType.LLM,
        addresses=[address],
        primary_address=address,
        public_key=public_key,
        created_at=datetime.now(),
        last_seen=datetime.now(),
    )

    # Convert CapabilityMesh Capability to A2A Agent Card
    agent_card = AgentCardConverter.capability_to_agent_card(
        capability, agent_identity
    )

    print("CapabilityMesh Capability converted to A2A Agent Card:")
    print(f"  Name: {agent_card['name']}")
    print(f"  Version: {agent_card['version']}")
    print(f"  Capabilities: {agent_card['capabilities']}")
    print(f"  URL: {agent_card['url']}")
    print(f"  Supported Protocols: {agent_card['supportedProtocols']}")

    # Wrap an A2A agent for CapabilityMesh discovery
    a2a_adapter = A2AAdapter.wrap(
        agent_url="http://example.com/a2a-agent",
        agent_card=agent_card
    )

    print(f"\nA2A Agent wrapped for CapabilityMesh discovery:")
    print(f"  Agent DID: {a2a_adapter.agent_identity.did}")
    print(f"  Supports A2A: {a2a_adapter.supports_a2a}")

    print("\n[SUCCESS] A2A integration complete!")


def example_crewai_integration():
    """Example 2: CrewAI Integration."""
    print_section("Example 2: CrewAI Integration")

    # Create a mock CrewAI agent (normally would import from crewai)
    class MockCrewAIAgent:
        def __init__(self, role, goal, backstory, tools=None):
            self.role = role
            self.goal = goal
            self.backstory = backstory
            self.tools = tools or []

    # Create CrewAI agent
    crew_agent = MockCrewAIAgent(
        role="Data Analyst",
        goal="Analyze data and provide insights",
        backstory="Expert data analyst with 10 years of experience in statistical analysis",
        tools=["pandas", "numpy", "matplotlib"]
    )

    # Wrap with CapabilityMesh
    acdp_crew_agent = ACDPCrewAIAgent.wrap(crew_agent)

    print(f"CrewAI agent wrapped for CapabilityMesh:")
    print(f"  Agent Name: {acdp_crew_agent.agent_identity.name}")
    print(f"  Agent DID: {acdp_crew_agent.agent_identity.did}")

    # Auto-extract capabilities
    capabilities = acdp_crew_agent.register_auto_capabilities()

    print(f"\nExtracted {len(capabilities)} capabilities:")
    for cap in capabilities:
        print(f"  - {cap.name}: {cap.description}")
        print(f"    Tags: {cap.semantic.tags}")

    # Dynamic crew formation
    crew_manager = DynamicCrew()
    crew_manager.add_agent(acdp_crew_agent)

    print(f"\nDynamic Crew Manager:")
    print(f"  Total agents: {len(crew_manager.discovered_agents)}")

    print("\n[SUCCESS] CrewAI integration complete!")


def example_autogen_integration():
    """Example 3: AutoGen Integration."""
    print_section("Example 3: AutoGen Integration")

    # Create a mock AutoGen agent
    class MockAutoGenAgent:
        def __init__(self, name, system_message, llm_config):
            self.name = name
            self.system_message = system_message
            self.llm_config = llm_config
            self._function_map = {}

    # Create AutoGen agent
    autogen_agent = MockAutoGenAgent(
        name="CodeReviewer",
        system_message="You are an expert code reviewer specializing in Python",
        llm_config={"model": "gpt-4", "temperature": 0.7}
    )

    # Wrap with CapabilityMesh
    acdp_autogen_agent = ACDPAutoGenAgent.wrap(autogen_agent)

    print(f"AutoGen agent wrapped for CapabilityMesh:")
    print(f"  Agent Name: {acdp_autogen_agent.agent_identity.name}")
    print(f"  Agent Type: {acdp_autogen_agent.agent_identity.agent_type}")

    # Extract capabilities
    capabilities = acdp_autogen_agent.register_auto_capabilities()

    print(f"\nExtracted capabilities:")
    for cap in capabilities:
        print(f"  - {cap.name}")
        print(f"    Description: {cap.description}")
        print(f"    Tags: {cap.semantic.tags}")

    # Dynamic group chat
    chat_manager = DynamicGroupChat()
    chat_manager.add_agent(acdp_autogen_agent)

    print(f"\nDynamic Group Chat:")
    print(f"  Total agents: {len(chat_manager.discovered_agents)}")

    print("\n[SUCCESS] AutoGen integration complete!")


def example_langgraph_integration():
    """Example 4: LangGraph Integration."""
    print_section("Example 4: LangGraph Integration")

    # Create discovery node
    discovery_node = ACDPDiscoveryNode(
        query="translation capability",
        max_results=5,
        min_trust_level=0.6
    )

    print(f"Created LangGraph Discovery Node:")
    print(f"  Query: {discovery_node.query}")
    print(f"  Max Results: {discovery_node.max_results}")
    print(f"  Min Trust Level: {discovery_node.min_trust_level}")

    # Simulate node execution
    state = {"task": "Translate document"}
    updated_state = discovery_node(state)

    print(f"\nNode execution (placeholder):")
    print(f"  Discovery Query: {updated_state.get('discovery_query')}")
    print(f"  Discovered Agents: {len(updated_state.get('discovered_agents', []))}")

    # Create a mock LangGraph
    class MockLangGraph:
        def __init__(self, name):
            self.name = name

    mock_graph = MockLangGraph(name="TranslationWorkflow")

    # Wrap as CapabilityMesh agent
    acdp_graph_agent = ACDPLangGraphAgent.wrap(
        graph=mock_graph,
        name="TranslationWorkflow",
        description="LangGraph workflow for translation tasks"
    )

    print(f"\nLangGraph wrapped as CapabilityMesh agent:")
    print(f"  Agent Name: {acdp_graph_agent.agent_identity.name}")
    print(f"  Description: {acdp_graph_agent.agent_identity.description}")

    capabilities = acdp_graph_agent.register_auto_capabilities()
    print(f"  Capabilities: {len(capabilities)}")

    print("\n[SUCCESS] LangGraph integration complete!")


def example_unified_workflow():
    """Example 5: Unified Multi-Framework Workflow."""
    print_section("Example 5: Unified Multi-Framework Workflow")

    print("Scenario: Coordinate agents from different frameworks via CapabilityMesh\n")

    # 1. CrewAI Translator
    class MockCrewAgent:
        def __init__(self, role, goal, backstory):
            self.role = role
            self.goal = goal
            self.backstory = backstory

    crew_translator = ACDPCrewAIAgent.wrap(
        MockCrewAgent(
            role="Translator",
            goal="Translate documents",
            backstory="Professional translator"
        )
    )
    crew_translator.register_auto_capabilities()

    # 2. AutoGen Analyst
    class MockAutoGenAgent:
        def __init__(self, name, system_message):
            self.name = name
            self.system_message = system_message
            self.llm_config = {}

    autogen_analyst = ACDPAutoGenAgent.wrap(
        MockAutoGenAgent(
            name="Analyst",
            system_message="Expert data analyst"
        )
    )
    autogen_analyst.register_auto_capabilities()

    # 3. A2A Writer (via agent card)
    a2a_writer = A2AAdapter.wrap(
        agent_url="http://example.com/writer",
        name="Writer",
        agent_card={
            "name": "Writer",
            "description": "Professional content writer",
            "version": "1.0.0",
            "capabilities": ["writing", "editing"],
            "url": "http://example.com/writer",
            "supportedProtocols": ["json-rpc"]
        }
    )
    a2a_writer.register_auto_capabilities()

    print("Agents registered:")
    print(f"  1. {crew_translator.agent_identity.name} (CrewAI)")
    print(f"     Capabilities: {[c.name for c in crew_translator.capabilities]}")

    print(f"  2. {autogen_analyst.agent_identity.name} (AutoGen)")
    print(f"     Capabilities: {[c.name for c in autogen_analyst.capabilities]}")

    print(f"  3. {a2a_writer.agent_identity.name} (A2A)")
    print(f"     Capabilities: {[c.name for c in a2a_writer.capabilities]}")

    print("\nWorkflow:")
    print("  1. Discover translator via CapabilityMesh")
    print("  2. Negotiate terms with translator")
    print("  3. Execute translation (via CrewAI)")
    print("  4. Discover analyst via CapabilityMesh")
    print("  5. Execute analysis (via AutoGen)")
    print("  6. Discover writer via CapabilityMesh")
    print("  7. Execute writing (via A2A protocol)")
    print("  8. Aggregate results")

    print("\n[SUCCESS] Framework coordination complete!")
    print("\nKey Insight:")
    print("  CapabilityMesh enables agents from different frameworks to:")
    print("  - Discover each other's capabilities")
    print("  - Negotiate collaboration terms")
    print("  - Execute tasks using their native frameworks")
    print("  - Work together seamlessly!")


def main():
    """Run all integration examples."""
    print("\n" + "*" * 70)
    print("  CapabilityMesh Multi-Framework Integration Examples")
    print("*" * 70)

    # Run examples
    example_a2a_integration()
    example_crewai_integration()
    example_autogen_integration()
    example_langgraph_integration()
    example_unified_workflow()

    print("\n" + "=" * 70)
    print("  *** All integration examples completed successfully! ***")
    print("=" * 70)

    print("\nNext Steps:")
    print("  1. Install frameworks: pip install capabilitymesh[frameworks]")
    print("  2. Replace mock agents with real framework instances")
    print("  3. Implement actual discovery and negotiation logic")
    print("  4. Deploy agents and test cross-framework collaboration")

    print("\nFramework Compatibility Matrix:")
    print("  [X] A2A (Google/Linux Foundation)")
    print("  [X] CrewAI (Multi-agent collaboration)")
    print("  [X] AutoGen (Microsoft conversational agents)")
    print("  [X] LangGraph (LangChain workflows)")
    print("  [ ] LlamaIndex (Planned)")
    print("  [ ] Haystack (Planned)")


if __name__ == "__main__":
    main()
