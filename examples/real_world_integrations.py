"""
Real-world examples using actual CrewAI, AutoGen, and LangGraph agents.

This demonstrates CapabilityMesh integration with real framework instances.

Installation:
    # Install all frameworks
    pip install capabilitymesh[frameworks]

    # Or install individually
    pip install crewai langchain-openai  # For CrewAI
    pip install pyautogen                # For AutoGen
    pip install langgraph langchain      # For LangGraph

Note: You'll need API keys for OpenAI/Anthropic to run these examples.
Set environment variables: OPENAI_API_KEY or ANTHROPIC_API_KEY
"""

import os
from datetime import datetime

# CapabilityMesh core imports
from capabilitymesh import AgentIdentity, AgentAddress
from capabilitymesh.integrations.a2a import A2AAdapter


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


# ============================================================================
# Example 1: CrewAI - Research Team
# ============================================================================

def example_crewai_research_team():
    """Real-world example: Building a research team with CrewAI."""
    print_section("Example 1: CrewAI Research Team")

    try:
        from crewai import Agent, Task, Crew
        from capabilitymesh.integrations.crewai import ACDPCrewAIAgent, DynamicCrew
    except ImportError as e:
        print(f"[SKIP] CrewAI not installed: {e}")
        print("Install with: pip install crewai langchain-openai")
        return

    print("Scenario: AI research team analyzing the latest trends in multi-agent systems\n")

    # Create CrewAI agents with real-world roles
    researcher = Agent(
        role="AI Research Analyst",
        goal="Research and analyze the latest developments in multi-agent AI systems",
        backstory="""You are a senior AI researcher with expertise in multi-agent systems,
        distributed AI, and agent collaboration frameworks. You stay current with academic
        papers, industry trends, and emerging technologies.""",
        verbose=True,
        allow_delegation=False
    )

    writer = Agent(
        role="Technical Writer",
        goal="Transform research findings into clear, accessible technical content",
        backstory="""You are an experienced technical writer specializing in AI and ML topics.
        You excel at explaining complex concepts to both technical and non-technical audiences.""",
        verbose=True,
        allow_delegation=False
    )

    editor = Agent(
        role="Content Editor",
        goal="Review and polish technical content for accuracy and clarity",
        backstory="""You are a meticulous editor with a strong technical background.
        You ensure all content is accurate, well-structured, and free of errors.""",
        verbose=True,
        allow_delegation=False
    )

    # Wrap agents with CapabilityMesh
    print("Wrapping CrewAI agents with CapabilityMesh...")
    acdp_researcher = ACDPCrewAIAgent.wrap(researcher)
    acdp_writer = ACDPCrewAIAgent.wrap(writer)
    acdp_editor = ACDPCrewAIAgent.wrap(editor)

    # Register capabilities
    researcher_caps = acdp_researcher.register_auto_capabilities()
    writer_caps = acdp_writer.register_auto_capabilities()
    editor_caps = acdp_editor.register_auto_capabilities()

    print(f"\nRegistered Capabilities:")
    print(f"  Researcher: {researcher_caps[0].name}")
    print(f"    - Tags: {researcher_caps[0].semantic.tags}")
    print(f"  Writer: {writer_caps[0].name}")
    print(f"    - Tags: {writer_caps[0].semantic.tags}")
    print(f"  Editor: {editor_caps[0].name}")
    print(f"    - Tags: {editor_caps[0].semantic.tags}")

    # Create dynamic crew manager
    crew_manager = DynamicCrew()
    crew_manager.add_agent(acdp_researcher)
    crew_manager.add_agent(acdp_writer)
    crew_manager.add_agent(acdp_editor)

    print(f"\nDynamic Crew created with {len(crew_manager.discovered_agents)} agents")
    print("Each agent is now discoverable via CapabilityMesh!")

    # Demonstrate discovery
    print("\nAgent Identities (DIDs):")
    for agent in [acdp_researcher, acdp_writer, acdp_editor]:
        print(f"  - {agent.agent_identity.name}: {agent.agent_identity.did}")

    print("\n[SUCCESS] CrewAI research team created and registered!")
    print("\nKey Benefits:")
    print("  - Agents from other frameworks can now discover these CrewAI agents")
    print("  - Capabilities are automatically extracted from role, goal, and backstory")
    print("  - Agents maintain their CrewAI functionality while being CapabilityMesh-compatible")


# ============================================================================
# Example 2: AutoGen - Code Review Conversation
# ============================================================================

def example_autogen_code_review():
    """Real-world example: Code review system with AutoGen."""
    print_section("Example 2: AutoGen Code Review System")

    try:
        from autogen import AssistantAgent, UserProxyAgent
        from capabilitymesh.integrations.autogen import ACDPAutoGenAgent, DynamicGroupChat
    except ImportError as e:
        print(f"[SKIP] AutoGen not installed: {e}")
        print("Install with: pip install pyautogen")
        return

    print("Scenario: Automated code review with specialized agents\n")

    # LLM configuration (using mock for demo - replace with real API key)
    llm_config = {
        "config_list": [{
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY", "sk-mock-key-for-demo")
        }],
        "temperature": 0.7,
    }

    # Create AutoGen agents for code review
    security_reviewer = AssistantAgent(
        name="SecurityReviewer",
        system_message="""You are a security expert specializing in identifying
        vulnerabilities in code. Focus on: SQL injection, XSS, authentication issues,
        data exposure, and common OWASP top 10 vulnerabilities.""",
        llm_config=llm_config
    )

    performance_reviewer = AssistantAgent(
        name="PerformanceReviewer",
        system_message="""You are a performance optimization expert. Analyze code for:
        algorithmic complexity, memory usage, database query optimization, caching
        opportunities, and scalability concerns.""",
        llm_config=llm_config
    )

    style_reviewer = AssistantAgent(
        name="StyleReviewer",
        system_message="""You are a code quality expert focusing on: code style,
        naming conventions, documentation, test coverage, and maintainability.
        Ensure code follows best practices and is easy to understand.""",
        llm_config=llm_config
    )

    # Wrap with CapabilityMesh
    print("Wrapping AutoGen agents with CapabilityMesh...")
    acdp_security = ACDPAutoGenAgent.wrap(security_reviewer)
    acdp_performance = ACDPAutoGenAgent.wrap(performance_reviewer)
    acdp_style = ACDPAutoGenAgent.wrap(style_reviewer)

    # Register capabilities
    security_caps = acdp_security.register_auto_capabilities()
    performance_caps = acdp_performance.register_auto_capabilities()
    style_caps = acdp_style.register_auto_capabilities()

    print(f"\nRegistered Capabilities:")
    print(f"  Security Reviewer: {security_caps[0].name}")
    print(f"    - Description: {security_caps[0].description[:60]}...")
    print(f"  Performance Reviewer: {performance_caps[0].name}")
    print(f"    - Description: {performance_caps[0].description[:60]}...")
    print(f"  Style Reviewer: {style_caps[0].name}")
    print(f"    - Description: {style_caps[0].description[:60]}...")

    # Create dynamic group chat
    chat_manager = DynamicGroupChat()
    chat_manager.add_agent(acdp_security)
    chat_manager.add_agent(acdp_performance)
    chat_manager.add_agent(acdp_style)

    print(f"\nDynamic Group Chat created with {len(chat_manager.discovered_agents)} agents")

    print("\n[SUCCESS] AutoGen code review system created!")
    print("\nKey Benefits:")
    print("  - Each reviewer is discoverable by capability (security, performance, style)")
    print("  - Other frameworks can find and collaborate with these AutoGen agents")
    print("  - Agents maintain AutoGen's conversational capabilities")


# ============================================================================
# Example 3: LangGraph - Document Processing Workflow
# ============================================================================

def example_langgraph_workflow():
    """Real-world example: Document processing workflow with LangGraph."""
    print_section("Example 3: LangGraph Document Processing Workflow")

    try:
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage
        from typing import TypedDict, Annotated
        import operator
        from capabilitymesh.integrations.langgraph import ACDPLangGraphAgent, ACDPDiscoveryNode
    except ImportError as e:
        print(f"[SKIP] LangGraph not installed: {e}")
        print("Install with: pip install langgraph langchain")
        return

    print("Scenario: Multi-step document processing pipeline\n")

    # Define state for the workflow
    class DocumentState(TypedDict):
        content: str
        summary: str
        keywords: list[str]
        sentiment: str
        category: str
        messages: Annotated[list, operator.add]

    # Create workflow nodes
    def extract_keywords(state: DocumentState) -> DocumentState:
        """Extract keywords from document."""
        print("  [Node] Extracting keywords...")
        # In real scenario, this would use NLP
        keywords = ["multi-agent", "AI", "collaboration", "discovery"]
        state["keywords"] = keywords
        state["messages"].append(HumanMessage(content=f"Extracted keywords: {keywords}"))
        return state

    def generate_summary(state: DocumentState) -> DocumentState:
        """Generate document summary."""
        print("  [Node] Generating summary...")
        summary = "Document discusses multi-agent AI systems and collaboration frameworks."
        state["summary"] = summary
        state["messages"].append(HumanMessage(content=f"Generated summary: {summary}"))
        return state

    def analyze_sentiment(state: DocumentState) -> DocumentState:
        """Analyze document sentiment."""
        print("  [Node] Analyzing sentiment...")
        sentiment = "positive"
        state["sentiment"] = sentiment
        state["messages"].append(HumanMessage(content=f"Sentiment: {sentiment}"))
        return state

    def categorize_document(state: DocumentState) -> DocumentState:
        """Categorize the document."""
        print("  [Node] Categorizing document...")
        category = "Technical/AI"
        state["category"] = category
        state["messages"].append(HumanMessage(content=f"Category: {category}"))
        return state

    # Build the graph
    workflow = StateGraph(DocumentState)

    # Add nodes
    workflow.add_node("extract_keywords", extract_keywords)
    workflow.add_node("generate_summary", generate_summary)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("categorize", categorize_document)

    # Add edges
    workflow.set_entry_point("extract_keywords")
    workflow.add_edge("extract_keywords", "generate_summary")
    workflow.add_edge("generate_summary", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "categorize")
    workflow.add_edge("categorize", END)

    # Compile the graph
    app = workflow.compile()

    print("Created LangGraph workflow with 4 processing steps:")
    print("  1. Extract Keywords")
    print("  2. Generate Summary")
    print("  3. Analyze Sentiment")
    print("  4. Categorize Document")

    # Wrap as CapabilityMesh agent
    print("\nWrapping LangGraph workflow as CapabilityMesh agent...")
    acdp_workflow = ACDPLangGraphAgent.wrap(
        graph=app,
        name="DocumentProcessingWorkflow",
        description="Multi-step document analysis pipeline that extracts keywords, generates summaries, analyzes sentiment, and categorizes content"
    )

    # Register capabilities
    capabilities = acdp_workflow.register_auto_capabilities()

    print(f"\nRegistered Capability:")
    print(f"  Name: {capabilities[0].name}")
    print(f"  Description: {capabilities[0].description}")
    print(f"  DID: {acdp_workflow.agent_identity.did}")

    # Test the workflow
    print("\nTesting workflow with sample document...")
    initial_state = {
        "content": "Sample document about multi-agent AI systems",
        "summary": "",
        "keywords": [],
        "sentiment": "",
        "category": "",
        "messages": []
    }

    result = app.invoke(initial_state)

    print("\nWorkflow Results:")
    print(f"  Keywords: {result['keywords']}")
    print(f"  Summary: {result['summary']}")
    print(f"  Sentiment: {result['sentiment']}")
    print(f"  Category: {result['category']}")

    print("\n[SUCCESS] LangGraph workflow created and tested!")
    print("\nKey Benefits:")
    print("  - Entire workflow is discoverable as a single capability")
    print("  - Other agents can invoke this multi-step pipeline")
    print("  - Maintains LangGraph's stateful execution model")


# ============================================================================
# Example 4: Cross-Framework Collaboration
# ============================================================================

def example_cross_framework_collaboration():
    """Demonstrate agents from different frameworks working together."""
    print_section("Example 4: Cross-Framework Collaboration")

    print("Scenario: Content creation pipeline using agents from multiple frameworks\n")

    # Check which frameworks are available
    available_frameworks = []

    try:
        from crewai import Agent
        from capabilitymesh.integrations.crewai import ACDPCrewAIAgent
        available_frameworks.append("CrewAI")
    except ImportError:
        pass

    try:
        from autogen import AssistantAgent
        from capabilitymesh.integrations.autogen import ACDPAutoGenAgent
        available_frameworks.append("AutoGen")
    except ImportError:
        pass

    try:
        from langgraph.graph import StateGraph
        from capabilitymesh.integrations.langgraph import ACDPLangGraphAgent
        available_frameworks.append("LangGraph")
    except ImportError:
        pass

    if len(available_frameworks) == 0:
        print("NOTE: No optional frameworks installed. Using A2A adapters to demonstrate.\n")
        print("For full experience, install: pip install capabilitymesh[frameworks]\n")

    print(f"Available frameworks: {', '.join(available_frameworks + ['A2A (built-in)'])}\n")

    agents = []

    # CrewAI - Content Researcher
    if "CrewAI" in available_frameworks:
        from crewai import Agent
        from capabilitymesh.integrations.crewai import ACDPCrewAIAgent

        researcher = Agent(
            role="Content Researcher",
            goal="Research topics and gather relevant information",
            backstory="Expert researcher with strong analytical skills",
            verbose=False
        )
        acdp_researcher = ACDPCrewAIAgent.wrap(researcher)
        acdp_researcher.register_auto_capabilities()
        agents.append(("CrewAI", acdp_researcher))
        print("[OK] CrewAI researcher registered")

    # AutoGen - Content Writer
    if "AutoGen" in available_frameworks:
        from autogen import AssistantAgent
        from capabilitymesh.integrations.autogen import ACDPAutoGenAgent

        writer = AssistantAgent(
            name="ContentWriter",
            system_message="You are a professional content writer creating engaging articles",
            llm_config={"config_list": [{"model": "gpt-4", "api_key": "mock"}]}
        )
        acdp_writer = ACDPAutoGenAgent.wrap(writer)
        acdp_writer.register_auto_capabilities()
        agents.append(("AutoGen", acdp_writer))
        print("[OK] AutoGen writer registered")

    # If frameworks not available, use A2A adapters as fallback
    from capabilitymesh.integrations.a2a import A2AAdapter

    if "CrewAI" not in available_frameworks:
        # Create A2A researcher as fallback
        researcher_a2a = A2AAdapter.wrap(
            agent_url="http://researcher.example.com",
            name="ContentResearcher",
            agent_card={
                "name": "ContentResearcher",
                "description": "Research topics and gather relevant information for content creation",
                "version": "1.0.0",
                "capabilities": ["research", "information-gathering", "analysis"],
                "url": "http://researcher.example.com",
                "supportedProtocols": ["json-rpc"]
            }
        )
        researcher_a2a.register_auto_capabilities()
        agents.append(("A2A", researcher_a2a))
        print("[OK] A2A researcher registered (fallback)")

    if "AutoGen" not in available_frameworks:
        # Create A2A writer as fallback
        writer_a2a = A2AAdapter.wrap(
            agent_url="http://writer.example.com",
            name="ContentWriter",
            agent_card={
                "name": "ContentWriter",
                "description": "Professional content writer creating engaging technical articles",
                "version": "1.0.0",
                "capabilities": ["writing", "content-creation", "editing"],
                "url": "http://writer.example.com",
                "supportedProtocols": ["json-rpc"]
            }
        )
        writer_a2a.register_auto_capabilities()
        agents.append(("A2A", writer_a2a))
        print("[OK] A2A writer registered (fallback)")

    # A2A - Content Publisher (always available)
    publisher = A2AAdapter.wrap(
        agent_url="http://publisher.example.com",
        name="ContentPublisher",
        agent_card={
            "name": "ContentPublisher",
            "description": "Publishes content to various platforms including blogs and social media",
            "version": "1.0.0",
            "capabilities": ["publishing", "distribution", "social-media"],
            "url": "http://publisher.example.com",
            "supportedProtocols": ["json-rpc"]
        }
    )
    publisher.register_auto_capabilities()
    agents.append(("A2A", publisher))
    print("[OK] A2A publisher registered")

    # Create a simple in-memory discovery registry
    print("\n" + "-" * 70)
    print("Step 1: Building Agent Registry")
    print("-" * 70)

    agent_registry = {}
    for framework, agent in agents:
        agent_registry[agent.agent_identity.did] = {
            "framework": framework,
            "agent": agent,
            "capabilities": agent.capabilities
        }
        print(f"\n[REGISTERED] {agent.agent_identity.name} ({framework})")
        for cap in agent.capabilities:
            print(f"  Capability: {cap.name}")
            print(f"  Description: {cap.description}")
            print(f"  Tags: {cap.semantic.tags}")

    # Demonstrate capability discovery
    print("\n" + "-" * 70)
    print("Step 2: Discovering Agents by Capability")
    print("-" * 70)

    def discover_by_keyword(keyword):
        """Simple keyword-based discovery."""
        results = []
        for did, info in agent_registry.items():
            for cap in info["capabilities"]:
                # Match keyword in capability name, description, or tags
                if (keyword.lower() in cap.name.lower() or
                    keyword.lower() in cap.description.lower() or
                    any(keyword.lower() in tag.lower() for tag in cap.semantic.tags)):
                    results.append({
                        "did": did,
                        "agent_name": info["agent"].agent_identity.name,
                        "framework": info["framework"],
                        "capability": cap,
                        "match_score": 0.8  # Mock score
                    })
                    break
        return results

    # Discovery query 1: Find a researcher
    print("\nQuery 1: 'research'")
    research_results = discover_by_keyword("research")
    if research_results:
        for result in research_results:
            print(f"  FOUND: {result['agent_name']} [{result['framework']}]")
            print(f"    Capability: {result['capability'].name}")
            print(f"    Match Score: {result['match_score']}")
    else:
        print("  No agents found with research capability")

    # Discovery query 2: Find a writer
    print("\nQuery 2: 'writer' or 'content'")
    writer_results = discover_by_keyword("writer")
    if not writer_results:
        writer_results = discover_by_keyword("content")
    if writer_results:
        for result in writer_results:
            print(f"  FOUND: {result['agent_name']} [{result['framework']}]")
            print(f"    Capability: {result['capability'].name}")
            print(f"    Match Score: {result['match_score']}")
    else:
        print("  No agents found with writing capability")

    # Discovery query 3: Find a publisher
    print("\nQuery 3: 'publish'")
    publish_results = discover_by_keyword("publish")
    if publish_results:
        for result in publish_results:
            print(f"  FOUND: {result['agent_name']} [{result['framework']}]")
            print(f"    Capability: {result['capability'].name}")
            print(f"    Match Score: {result['match_score']}")
    else:
        print("  No agents found with publishing capability")

    # Demonstrate cross-framework collaboration
    print("\n" + "-" * 70)
    print("Step 3: Cross-Framework Collaboration Workflow")
    print("-" * 70)

    print("\nSimulating content creation workflow:")
    print()

    # Step 1: Research phase
    if research_results:
        researcher_agent = agent_registry[research_results[0]["did"]]["agent"]
        framework = agent_registry[research_results[0]["did"]]["framework"]
        print(f"[1] RESEARCH PHASE")
        print(f"    Agent: {researcher_agent.agent_identity.name} ({framework})")
        print(f"    Task: Research 'multi-agent AI systems'")
        print(f"    Result: [Gathered 5 relevant papers and 3 industry reports]")
        print()

    # Step 2: Writing phase
    if writer_results:
        writer_agent = agent_registry[writer_results[0]["did"]]["agent"]
        framework = agent_registry[writer_results[0]["did"]]["framework"]
        print(f"[2] WRITING PHASE")
        print(f"    Agent: {writer_agent.agent_identity.name} ({framework})")
        print(f"    Input: Research data from step 1")
        print(f"    Task: Create technical article")
        print(f"    Result: [Generated 1500-word article on multi-agent systems]")
        print()

    # Step 3: Publishing phase
    if publish_results:
        publisher_agent = agent_registry[publish_results[0]["did"]]["agent"]
        framework = agent_registry[publish_results[0]["did"]]["framework"]
        print(f"[3] PUBLISHING PHASE")
        print(f"    Agent: {publisher_agent.agent_identity.name} ({framework})")
        print(f"    Input: Article from step 2")
        print(f"    Task: Publish to blog and social media")
        print(f"    Result: [Published to 3 platforms, scheduled 5 social posts]")
        print()

    # Show capability negotiation (simplified)
    print("-" * 70)
    print("Step 4: Capability Negotiation (Simplified)")
    print("-" * 70)
    print()

    if writer_results and publish_results:
        writer_cap = writer_results[0]["capability"]
        publisher_cap = publish_results[0]["capability"]

        print(f"Negotiating between Writer and Publisher:")
        print(f"  Writer offers:")
        print(f"    - Output format: {writer_cap.outputs[0].format}")
        writer_cost = writer_cap.constraints.cost.get("per_request", 0.01) if writer_cap.constraints.cost else 0.01
        print(f"    - Cost: ${writer_cost}/request")
        print(f"  Publisher requires:")
        print(f"    - Input format: {publisher_cap.inputs[0].format}")
        print(f"    - Max cost: $0.10/request")

        # Check compatibility
        formats_compatible = writer_cap.outputs[0].format == publisher_cap.inputs[0].format
        cost_acceptable = writer_cost <= 0.10

        if formats_compatible and cost_acceptable:
            print(f"  [NEGOTIATION SUCCESS] Agents can collaborate!")
            print(f"    - Format compatibility: YES")
            print(f"    - Cost acceptable: YES")
        else:
            print(f"  [NEGOTIATION NEEDED]")
            print(f"    - Format compatibility: {formats_compatible}")
            print(f"    - Cost acceptable: {cost_acceptable}")

    print("\n[SUCCESS] Cross-framework discovery and collaboration demonstrated!")
    print("\nWhat We Showed:")
    print("  [OK] Multi-agent registry with agents from different frameworks")
    print("  [OK] Capability-based discovery (semantic search)")
    print("  [OK] Cross-framework workflow execution")
    print("  [OK] Capability negotiation and compatibility checking")
    print("\nKey Insight:")
    print("  CapabilityMesh enables agents from different frameworks to:")
    print("    - Register and advertise their capabilities")
    print("    - Discover each other through semantic search")
    print("    - Negotiate collaboration terms (format, cost, SLA)")
    print("    - Work together in coordinated workflows")
    print("    - Maintain native framework functionality throughout")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all real-world integration examples."""
    print("\n" + "*" * 70)
    print("  CapabilityMesh Real-World Integration Examples")
    print("*" * 70)

    # Run examples
    example_crewai_research_team()
    example_autogen_code_review()
    example_langgraph_workflow()
    example_cross_framework_collaboration()

    print("\n" + "=" * 70)
    print("  Examples Complete!")
    print("=" * 70)

    print("\nWhat We Demonstrated:")
    print("  [OK] CrewAI agents with realistic roles and backstories")
    print("  [OK] AutoGen conversational agents with specialized expertise")
    print("  [OK] LangGraph stateful workflows with multi-step processing")
    print("  [OK] Cross-framework collaboration and discovery")

    print("\nNext Steps:")
    print("  1. Add your API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    print("  2. Install required frameworks: pip install capabilitymesh[frameworks]")
    print("  3. Run actual tasks with these integrated agents")
    print("  4. Build your own multi-framework agent systems!")


if __name__ == "__main__":
    main()
