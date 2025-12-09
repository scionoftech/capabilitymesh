"""Base adapter interface for framework integrations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from capabilitymesh.core.identity import AgentIdentity
from capabilitymesh.schemas.capability import Capability


class FrameworkAdapter(ABC):
    """Base class for framework adapters.

    All framework integrations (CrewAI, AutoGen, LangGraph, etc.) should
    inherit from this class and implement the required methods.
    """

    def __init__(self, agent_identity: AgentIdentity, framework_agent: Any) -> None:
        """Initialize the adapter.

        Args:
            agent_identity: ACDP agent identity
            framework_agent: The underlying framework-specific agent
        """
        self.agent_identity = agent_identity
        self.framework_agent = framework_agent
        self.capabilities: List[Capability] = []
        self.supports_a2a = False

    @classmethod
    @abstractmethod
    def wrap(
        cls,
        framework_agent: Any,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> "FrameworkAdapter":
        """Wrap an existing framework agent with ACDP capabilities.

        Args:
            framework_agent: The framework-specific agent to wrap
            name: Optional name for the ACDP agent
            **kwargs: Additional framework-specific arguments

        Returns:
            FrameworkAdapter instance
        """
        pass

    @abstractmethod
    def extract_capabilities(self) -> List[Capability]:
        """Auto-extract capabilities from the framework agent.

        Each framework adapter should implement logic to automatically
        discover what the wrapped agent can do and convert it to ACDP
        capabilities.

        Returns:
            List of extracted capabilities
        """
        pass

    @abstractmethod
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the underlying framework.

        Args:
            task: Task specification (framework-specific format)

        Returns:
            Task result (framework-specific format)
        """
        pass

    def register_capability(self, capability: Capability) -> None:
        """Register a capability manually.

        Args:
            capability: Capability to register
        """
        if capability not in self.capabilities:
            self.capabilities.append(capability)

    def register_auto_capabilities(self) -> List[Capability]:
        """Auto-extract and register capabilities.

        Returns:
            List of registered capabilities
        """
        extracted = self.extract_capabilities()
        for cap in extracted:
            self.register_capability(cap)
        return extracted

    def get_capabilities(self) -> List[Capability]:
        """Get all registered capabilities.

        Returns:
            List of capabilities
        """
        return self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        """Convert adapter to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {
            "agent_identity": self.agent_identity.model_dump(),
            "framework": self.__class__.__name__,
            "capabilities": [cap.to_summary() for cap in self.capabilities],
            "supports_a2a": self.supports_a2a,
        }


class A2ACompatibleAdapter(FrameworkAdapter):
    """Base class for adapters that support A2A protocol.

    Frameworks that implement A2A protocol should inherit from this class.
    """

    def __init__(self, agent_identity: AgentIdentity, framework_agent: Any) -> None:
        """Initialize the A2A-compatible adapter."""
        super().__init__(agent_identity, framework_agent)
        self.supports_a2a = True

    @abstractmethod
    def to_agent_card(self) -> Dict[str, Any]:
        """Convert ACDP capabilities to A2A Agent Card format.

        Returns:
            A2A Agent Card as dictionary
        """
        pass

    @abstractmethod
    def execute_a2a_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using A2A protocol.

        Args:
            task: A2A task specification

        Returns:
            A2A task result
        """
        pass
