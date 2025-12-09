"""CrewAI integration for ACDP.

This module provides integration with CrewAI, allowing CrewAI agents to:
- Be discoverable via ACDP
- Advertise their capabilities
- Participate in ACDP-based multi-agent coordination

CrewAI: https://github.com/joaomdmoura/crewAI
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


class ACDPCrewAIAgent(FrameworkAdapter):
    """ACDP adapter for CrewAI agents.

    Wraps CrewAI Agent instances to make them discoverable via ACDP.

    Example:
        ```python
        from crewai import Agent
        from capabilitymesh.integrations.crewai import ACDPCrewAIAgent

        # Create CrewAI agent
        crew_agent = Agent(
            role="Translator",
            goal="Translate text accurately",
            backstory="Expert translator with 10 years experience",
            tools=[translation_tool]
        )

        # Wrap with ACDP
        acdp_agent = ACDPCrewAIAgent.wrap(crew_agent)
        acdp_agent.register_auto_capabilities()
        acdp_agent.start_discovery()
        ```
    """

    def __init__(
        self,
        agent_identity: AgentIdentity,
        crew_agent: Any
    ) -> None:
        """Initialize CrewAI ACDP adapter.

        Args:
            agent_identity: ACDP agent identity
            crew_agent: CrewAI Agent instance
        """
        super().__init__(agent_identity, crew_agent)
        self.crew_agent = crew_agent

    @classmethod
    def wrap(
        cls,
        crew_agent: Any,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> "ACDPCrewAIAgent":
        """Wrap a CrewAI agent with ACDP capabilities.

        Args:
            crew_agent: CrewAI Agent instance
            name: Optional name (uses agent.role if not provided)
            **kwargs: Additional arguments

        Returns:
            ACDPCrewAIAgent instance
        """
        # Extract name from CrewAI agent
        if not name:
            # CrewAI agents have a 'role' attribute
            name = getattr(crew_agent, 'role', 'CrewAI Agent')

        # Create agent address (placeholder - in production, would be actual endpoint)
        address = AgentAddress(
            protocol="http",
            host="localhost",
            port=8000,
            path=f"/agents/{name.lower().replace(' ', '-')}"
        )

        # Generate identity
        public_key = f"-----BEGIN PUBLIC KEY-----\n{uuid4().hex}\n-----END PUBLIC KEY-----"
        did = AgentIdentity.generate_did(public_key)

        # Extract description from backstory
        description = getattr(crew_agent, 'backstory', f"CrewAI agent: {name}")

        agent_identity = AgentIdentity(
            did=did,
            name=name,
            agent_type=AgentType.LLM,  # CrewAI agents are typically LLM-based
            description=description[:200] if description else None,  # Truncate if too long
            addresses=[address],
            primary_address=address,
            public_key=public_key,
            created_at=datetime.now(),
            last_seen=datetime.now(),
        )

        return cls(agent_identity=agent_identity, crew_agent=crew_agent)

    def extract_capabilities(self) -> List[Capability]:
        """Extract capabilities from CrewAI agent.

        CrewAI agents have:
        - role: What they do
        - goal: Their objective
        - tools: Available tools
        - backstory: Context about expertise

        Returns:
            List of extracted ACDP Capabilities
        """
        capabilities = []

        # Extract basic information
        role = getattr(self.crew_agent, 'role', 'unknown')
        goal = getattr(self.crew_agent, 'goal', 'No goal specified')
        backstory = getattr(self.crew_agent, 'backstory', '')
        tools = getattr(self.crew_agent, 'tools', [])

        # Create main capability based on role
        capability_id = f"cap-crewai-{role.lower().replace(' ', '-')}-{uuid4().hex[:8]}"

        capability = Capability(
            id=capability_id,
            name=role.lower().replace(' ', '-'),
            description=f"{role}: {goal}",
            version=CapabilityVersion(major=1, minor=0, patch=0),
            capability_type=CapabilityType.UNSTRUCTURED,
            agent_type=AgentType.LLM,
            inputs=[
                CapabilityInputOutput(
                    format=IOFormat.TEXT,
                    description=f"Task description for {role}",
                    examples=[goal]
                )
            ],
            outputs=[
                CapabilityInputOutput(
                    format=IOFormat.TEXT,
                    description=f"Result from {role}",
                )
            ],
            unstructured_spec=UnstructuredCapability(
                system_prompt=backstory,
                model_info={"framework": "crewai", "role": role},
            ),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Add semantic metadata
        capability.semantic.tags = self._extract_tags_from_role(role)
        capability.semantic.categories = ["crewai", "agent", "llm"]

        # Add tool information to metadata
        if tools:
            tool_names = [getattr(tool, 'name', str(tool)) for tool in tools]
            capability.semantic.tags.extend(tool_names)

        capabilities.append(capability)

        return capabilities

    def _extract_tags_from_role(self, role: str) -> List[str]:
        """Extract semantic tags from role description.

        Args:
            role: Agent role string

        Returns:
            List of tags
        """
        # Simple keyword extraction
        # In production, could use NLP for better extraction
        role_lower = role.lower()
        tags = [role_lower]

        # Add common keywords
        keywords = [
            "translator", "analyst", "writer", "researcher",
            "developer", "designer", "manager", "coordinator"
        ]

        for keyword in keywords:
            if keyword in role_lower:
                tags.append(keyword)

        return list(set(tags))

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the CrewAI agent.

        Args:
            task: Task specification with 'description' field

        Returns:
            Task result
        """
        # Extract task description
        task_description = task.get("description", task.get("instructions", ""))

        # CrewAI agents have an execute_task method
        # This is a placeholder - actual implementation depends on CrewAI version
        try:
            # CrewAI task execution (simplified)
            # In production: result = self.crew_agent.execute_task(task_description)
            result = {
                "status": "completed",
                "result": f"Task executed by {self.agent_identity.name}",
                "message": "CrewAI integration requires crewai package",
            }
            return result
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def start_discovery(self) -> None:
        """Start ACDP discovery for this agent.

        This would connect to the ACDP discovery network and advertise
        this agent's capabilities.
        """
        # Placeholder - would integrate with ACDP discovery engine
        # Discovery engine will be implemented in v0.2.0
        pass


class DynamicCrew:
    """Dynamically form CrewAI crews via ACDP discovery.

    Example:
        ```python
        crew_manager = DynamicCrew()

        # Discover agents for specific roles
        translator = crew_manager.discover_agent(query="translation")
        analyst = crew_manager.discover_agent(query="data analysis")

        # Form crew with discovered agents
        crew = crew_manager.form_crew([translator, analyst])
        result = crew.execute_task("Analyze and translate report")
        ```
    """

    def __init__(self) -> None:
        """Initialize dynamic crew manager."""
        self.discovered_agents: List[ACDPCrewAIAgent] = []

    def discover_agent(self, query: str) -> Optional[ACDPCrewAIAgent]:
        """Discover an agent via ACDP.

        Args:
            query: Discovery query

        Returns:
            Discovered ACDPCrewAIAgent or None
        """
        # Placeholder - would use ACDP discovery engine
        # Discovery engine will be implemented in v0.2.0
        return None

    def form_crew(self, agents: List[ACDPCrewAIAgent]) -> Dict[str, Any]:
        """Form a crew from discovered agents.

        Args:
            agents: List of ACDP CrewAI agents

        Returns:
            Crew configuration
        """
        crew_config = {
            "agents": [agent.crew_agent for agent in agents],
            "capabilities": sum([agent.capabilities for agent in agents], []),
        }
        return crew_config

    def add_agent(self, agent: ACDPCrewAIAgent) -> None:
        """Add an agent to the pool of discovered agents.

        Args:
            agent: ACDP CrewAI agent
        """
        self.discovered_agents.append(agent)
