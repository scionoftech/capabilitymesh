"""A2A client for executing tasks via the Agent2Agent protocol.

This client handles task creation, execution, and result retrieval using
the A2A JSON-RPC protocol over HTTP.
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from capabilitymesh.core.identity import AgentAddress


class A2AClient:
    """Client for interacting with agents via A2A protocol.

    This client implements the A2A protocol for task execution, following
    the specification at https://a2a-protocol.org/latest/specification/
    """

    def __init__(
        self,
        agent_url: str,
        timeout: int = 300,
        http_client: Optional[Any] = None
    ) -> None:
        """Initialize A2A client.

        Args:
            agent_url: URL of the A2A-compatible agent
            timeout: Request timeout in seconds
            http_client: Optional HTTP client (e.g., httpx.AsyncClient)
        """
        self.agent_url = agent_url
        self.timeout = timeout
        self.http_client = http_client  # Will be implemented with httpx

    @classmethod
    def from_agent_address(cls, address: AgentAddress, **kwargs: Any) -> "A2AClient":
        """Create A2A client from AgentAddress.

        Args:
            address: ACDP AgentAddress
            **kwargs: Additional arguments for A2AClient

        Returns:
            A2AClient instance
        """
        return cls(agent_url=address.to_uri(), **kwargs)

    @classmethod
    def from_acdp_agreement(cls, agreement: Dict[str, Any]) -> "A2AClient":
        """Create A2A client from ACDP negotiation agreement.

        Args:
            agreement: ACDP negotiation agreement

        Returns:
            A2AClient instance
        """
        # Extract agent URL from agreement
        agent_url = agreement.get("provider_address", "")
        return cls(agent_url=agent_url)

    def create_task(
        self,
        instructions: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new A2A task.

        Args:
            instructions: Task instructions
            task_id: Optional task ID (generated if not provided)
            metadata: Optional task metadata

        Returns:
            A2A task object
        """
        if not task_id:
            task_id = f"task-{uuid4().hex[:12]}"

        task = {
            "taskId": task_id,
            "instructions": instructions,
            "status": "pending",
            "metadata": metadata or {},
        }

        return task

    async def execute_task(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task via A2A protocol.

        Args:
            task: A2A task object

        Returns:
            Task result

        Note:
            This is a placeholder. Full implementation requires:
            1. JSON-RPC request formatting
            2. HTTP POST to agent_url
            3. Task status polling
            4. Result retrieval
        """
        # Placeholder implementation
        # Full implementation will use httpx for async HTTP
        raise NotImplementedError(
            "A2A task execution requires httpx dependency. "
            "Install with: pip install acdp[a2a]"
        )

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task.

        Args:
            task_id: Task ID

        Returns:
            Task status object
        """
        raise NotImplementedError("A2A client requires httpx dependency")

    async def wait_for_completion(
        self,
        task_id: str,
        poll_interval: int = 2
    ) -> Dict[str, Any]:
        """Wait for a task to complete.

        Args:
            task_id: Task ID
            poll_interval: Polling interval in seconds

        Returns:
            Completed task result
        """
        raise NotImplementedError("A2A client requires httpx dependency")

    def get_agent_card(self) -> Dict[str, Any]:
        """Retrieve the agent's Agent Card.

        Returns:
            Agent Card dictionary
        """
        raise NotImplementedError("A2A client requires httpx dependency")


class A2ATaskBuilder:
    """Builder for constructing A2A tasks."""

    def __init__(self) -> None:
        """Initialize task builder."""
        self.task: Dict[str, Any] = {
            "taskId": f"task-{uuid4().hex[:12]}",
            "status": "pending",
            "metadata": {},
        }

    def with_instructions(self, instructions: str) -> "A2ATaskBuilder":
        """Set task instructions.

        Args:
            instructions: Task instructions

        Returns:
            Self for chaining
        """
        self.task["instructions"] = instructions
        return self

    def with_task_id(self, task_id: str) -> "A2ATaskBuilder":
        """Set task ID.

        Args:
            task_id: Task ID

        Returns:
            Self for chaining
        """
        self.task["taskId"] = task_id
        return self

    def with_metadata(self, key: str, value: Any) -> "A2ATaskBuilder":
        """Add metadata to task.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self.task["metadata"][key] = value
        return self

    def with_context(self, context: Dict[str, Any]) -> "A2ATaskBuilder":
        """Add context to task.

        Args:
            context: Context dictionary

        Returns:
            Self for chaining
        """
        self.task["context"] = context
        return self

    def build(self) -> Dict[str, Any]:
        """Build the task object.

        Returns:
            A2A task dictionary
        """
        if "instructions" not in self.task:
            raise ValueError("Task must have instructions")
        return self.task
