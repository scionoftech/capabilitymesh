"""A2A (Agent2Agent) protocol integration for ACDP.

This module provides bidirectional integration between ACDP and Google's
Agent2Agent (A2A) protocol:

- Convert ACDP Capabilities to A2A Agent Cards
- Execute tasks via A2A protocol
- Wrap A2A agents for ACDP discovery
"""

from capabilitymesh.integrations.a2a.converter import (
    AgentCardConverter,
    capability_to_agent_card,
    agent_card_to_capability,
)
from capabilitymesh.integrations.a2a.client import A2AClient
from capabilitymesh.integrations.a2a.adapter import A2AAdapter, A2ADiscoveryBridge

__all__ = [
    "AgentCardConverter",
    "capability_to_agent_card",
    "agent_card_to_capability",
    "A2AClient",
    "A2AAdapter",
    "A2ADiscoveryBridge",
]
