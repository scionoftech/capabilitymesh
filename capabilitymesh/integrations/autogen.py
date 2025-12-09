"""AutoGen integration for ACDP.

This module provides integration with Microsoft's AutoGen framework for
multi-agent conversations with ACDP-based discovery.

AutoGen: https://github.com/microsoft/autogen
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from capabilitymesh.core.identity import AgentIdentity, AgentAddress
from capabilitymesh.core.types import AgentType, CapabilityType, IOFormat
from capabilitymesh.schemas.capability import (
    Capability,
    CapabilityVersion,
    CapabilityInputOutput,
    UnstructuredCapability,
)
from capabilitymesh.integrations.base import FrameworkAdapter


class ACDPAutoGenAgent(FrameworkAdapter):
    """ACDP adapter for AutoGen ConversableAgent.

    Wraps AutoGen agents to make them discoverable via ACDP.

    Example:
        ```python
        from autogen import ConversableAgent
        from capabilitymesh.integrations.autogen import ACDPAutoGenAgent

        # Create AutoGen agent
        autogen_agent = ConversableAgent(
            name="DataAnalyst",
            system_message="You are a data analyst expert",
            llm_config={"model": "gpt-4"}
        )

        # Wrap with ACDP
        acdp_agent = ACDPAutoGenAgent.wrap(autogen_agent)
        acdp_agent.register_auto_capabilities()
        ```
    """

    def __init__(
        self,
        agent_identity: AgentIdentity,
        autogen_agent: Any
    ) -> None:
        """Initialize AutoGen ACDP adapter.

        Args:
            agent_identity: ACDP agent identity
            autogen_agent: AutoGen ConversableAgent instance
        """
        super().__init__(agent_identity, autogen_agent)
        self.autogen_agent = autogen_agent

    @classmethod
    def wrap(
        cls,
        autogen_agent: Any,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> "ACDPAutoGenAgent":
        """Wrap an AutoGen agent with ACDP capabilities.

        Args:
            autogen_agent: AutoGen ConversableAgent instance
            name: Optional name (uses agent.name if not provided)
            **kwargs: Additional arguments

        Returns:
            ACDPAutoGenAgent instance
        """
        # Extract name from AutoGen agent
        if not name:
            name = getattr(autogen_agent, 'name', 'AutoGen Agent')

        # Create agent address
        address = AgentAddress(
            protocol="http",
            host="localhost",
            port=8000,
            path=f"/autogen/{name.lower().replace(' ', '-')}"
        )

        # Generate identity
        public_key = f"-----BEGIN PUBLIC KEY-----\n{uuid4().hex}\n-----END PUBLIC KEY-----"
        did = AgentIdentity.generate_did(public_key)

        # Extract system message as description
        system_message = getattr(autogen_agent, 'system_message', '')
        description = system_message[:200] if system_message else f"AutoGen agent: {name}"

        agent_identity = AgentIdentity(
            did=did,
            name=name,
            agent_type=AgentType.LLM,  # AutoGen agents are typically LLM-based
            description=description,
            addresses=[address],
            primary_address=address,
            public_key=public_key,
            created_at=datetime.now(),
            last_seen=datetime.now(),
        )

        return cls(agent_identity=agent_identity, autogen_agent=autogen_agent)

    def extract_capabilities(self) -> List[Capability]:
        """Extract capabilities from AutoGen agent.

        AutoGen agents have:
        - name: Agent identifier
        - system_message: Description of role/expertise
        - llm_config: Model configuration
        - registered functions: Tool capabilities

        Returns:
            List of extracted ACDP Capabilities
        """
        capabilities = []

        name = getattr(self.autogen_agent, 'name', 'unknown')
        system_message = getattr(self.autogen_agent, 'system_message', '')
        llm_config = getattr(self.autogen_agent, 'llm_config', {})

        # Create main capability
        capability_id = f"cap-autogen-{name.lower().replace(' ', '-')}-{uuid4().hex[:8]}"

        capability = Capability(
            id=capability_id,
            name=name.lower().replace(' ', '-'),
            description=system_message or f"AutoGen agent: {name}",
            version=CapabilityVersion(major=1, minor=0, patch=0),
            capability_type=CapabilityType.UNSTRUCTURED,
            agent_type=AgentType.LLM,
            inputs=[
                CapabilityInputOutput(
                    format=IOFormat.TEXT,
                    description=f"Message for {name}",
                )
            ],
            outputs=[
                CapabilityInputOutput(
                    format=IOFormat.TEXT,
                    description=f"Response from {name}",
                )
            ],
            unstructured_spec=UnstructuredCapability(
                system_prompt=system_message,
                model_info={
                    "framework": "autogen",
                    "config": llm_config
                },
            ),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Extract tags from system message and name
        capability.semantic.tags = self._extract_tags(name, system_message)
        capability.semantic.categories = ["autogen", "agent", "llm", "conversation"]

        # Add function/tool capabilities
        # AutoGen agents can have registered functions
        function_map = getattr(self.autogen_agent, '_function_map', {})
        if function_map:
            capability.semantic.tags.extend(function_map.keys())

        capabilities.append(capability)

        return capabilities

    def _extract_tags(self, name: str, system_message: str) -> List[str]:
        """Extract tags from name and system message.

        Args:
            name: Agent name
            system_message: System message

        Returns:
            List of tags
        """
        tags = [name.lower()]

        # Extract keywords from system message
        keywords = ["analyst", "developer", "expert", "assistant", "coder", "writer"]
        text = (name + " " + system_message).lower()

        for keyword in keywords:
            if keyword in text:
                tags.append(keyword)

        return list(set(tags))

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the AutoGen agent.

        Args:
            task: Task specification with 'message' field

        Returns:
            Task result with agent's response
        """
        message = task.get("message", task.get("description", ""))

        try:
            # AutoGen message handling
            # In production: self.autogen_agent.receive(message, sender_agent)
            result = {
                "status": "completed",
                "response": f"Response from {self.agent_identity.name}",
                "message": "AutoGen integration requires autogen package",
            }
            return result
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


class DynamicGroupChat:
    """Dynamically form AutoGen group chats via ACDP discovery.

    Example:
        ```python
        chat_manager = DynamicGroupChat()

        # Discover agents for the task
        analyst = chat_manager.discover_and_add("data analysis")
        coder = chat_manager.discover_and_add("python coding")
        reviewer = chat_manager.discover_and_add("code review")

        # Start group chat
        chat = chat_manager.create_group_chat()
        result = chat.initiate_chat("Analyze this dataset and write code")
        ```
    """

    def __init__(self) -> None:
        """Initialize dynamic group chat manager."""
        self.discovered_agents: List[ACDPAutoGenAgent] = []

    def discover_and_add(self, query: str) -> Optional[ACDPAutoGenAgent]:
        """Discover an agent and add to group chat.

        Args:
            query: Discovery query for required capability

        Returns:
            Discovered ACDPAutoGenAgent or None
        """
        # Placeholder - would use ACDP discovery engine
        # Discovery engine will be implemented in v0.2.0
        return None

    def add_agent(self, agent: ACDPAutoGenAgent) -> None:
        """Add an agent to the group chat.

        Args:
            agent: ACDP AutoGen agent
        """
        self.discovered_agents.append(agent)

    def create_group_chat(self) -> Dict[str, Any]:
        """Create an AutoGen GroupChat with discovered agents.

        Returns:
            GroupChat configuration
        """
        return {
            "agents": [agent.autogen_agent for agent in self.discovered_agents],
            "capabilities": sum([agent.capabilities for agent in self.discovered_agents], []),
        }


class CapabilityBasedRouter:
    """Route messages to agents based on capabilities.

    Analyzes incoming messages and routes them to agents with
    appropriate capabilities discovered via ACDP.

    Example:
        ```python
        router = CapabilityBasedRouter()

        # Register agents
        router.register_agent(translator_agent)
        router.register_agent(analyst_agent)

        # Route message
        agent = router.route("Translate this report")
        # Returns translator_agent

        agent = router.route("Analyze this data")
        # Returns analyst_agent
        ```
    """

    def __init__(self) -> None:
        """Initialize capability-based router."""
        self.agents: List[ACDPAutoGenAgent] = []

    def register_agent(self, agent: ACDPAutoGenAgent) -> None:
        """Register an agent with the router.

        Args:
            agent: ACDP AutoGen agent
        """
        self.agents.append(agent)

    def route(self, message: str) -> Optional[ACDPAutoGenAgent]:
        """Route message to most appropriate agent.

        Args:
            message: Message to route

        Returns:
            Best matching agent or None
        """
        # Placeholder - would use semantic matching
        # Would compare message with agent capabilities
        # Semantic matching will be implemented in v0.5.0

        if not self.agents:
            return None

        # Simple fallback: return first agent
        return self.agents[0]

    def route_with_score(self, message: str) -> List[tuple[ACDPAutoGenAgent, float]]:
        """Route message and return agents with confidence scores.

        Args:
            message: Message to route

        Returns:
            List of (agent, score) tuples, sorted by score
        """
        # Placeholder - would use ACDP semantic matching
        results = []

        for agent in self.agents:
            # Calculate match score based on capabilities
            score = 0.5  # Placeholder

            results.append((agent, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results
