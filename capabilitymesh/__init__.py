"""
CapabilityMesh - The Capability Mesh for Multi-Agent Systems

CapabilityMesh is the first and only Python package providing universal capability
discovery and negotiation across all major agent frameworks (A2A, CrewAI, AutoGen, LangGraph),
with built-in trust and reputation management.

Key Features:
- Discover agents across any framework
- Negotiate collaboration terms (cost, SLA, constraints)
- Build trust through reputation tracking
- A2A protocol compatible
- Protocol-agnostic communication
"""

__version__ = "1.0.0-alpha.2"
__title__ = "CapabilityMesh"
__description__ = "Universal capability discovery and negotiation for multi-agent systems"
__author__ = "Sai Kumar Yava"
__author_email__ = "saikumar.geek@github.com"
__license__ = "Apache-2.0"
__url__ = "https://github.com/scionoftech/capabilitymesh"

# Core API
from capabilitymesh.mesh import Mesh, AgentInfo

# Identity and Types
from capabilitymesh.core.identity import AgentIdentity, AgentAddress
from capabilitymesh.core.types import AgentType, CapabilityType, IOFormat

# Capability Schemas
from capabilitymesh.schemas.capability import (
    Capability,
    CapabilityVersion,
    CapabilityInputOutput,
    StructuredCapability,
    UnstructuredCapability,
    CapabilityConstraints,
    SemanticMetadata,
)

# Storage
from capabilitymesh.storage import Storage, InMemoryStorage, AgentRecord

# Embeddings
from capabilitymesh.embeddings import (
    Embedder,
    KeywordEmbedder,
    auto_select_embedder,
)

# Trust
from capabilitymesh.trust import (
    SimpleTrustManager,
    TrustLevel,
    TrustScore,
)

__all__ = [
    # Core API
    "Mesh",
    "AgentInfo",
    # Identity
    "AgentIdentity",
    "AgentAddress",
    # Types
    "AgentType",
    "CapabilityType",
    "IOFormat",
    # Capability Schemas
    "Capability",
    "CapabilityVersion",
    "CapabilityInputOutput",
    "StructuredCapability",
    "UnstructuredCapability",
    "CapabilityConstraints",
    "SemanticMetadata",
    # Storage
    "Storage",
    "InMemoryStorage",
    "AgentRecord",
    # Embeddings
    "Embedder",
    "KeywordEmbedder",
    "auto_select_embedder",
    # Trust
    "SimpleTrustManager",
    "TrustLevel",
    "TrustScore",
]
