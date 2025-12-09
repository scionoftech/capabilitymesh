"""Core type definitions and enums for ACDP."""

from enum import Enum


class CapabilityType(str, Enum):
    """Types of capabilities an agent can provide."""

    STRUCTURED = "structured"      # API-based, deterministic
    UNSTRUCTURED = "unstructured"  # LLM-based, natural language
    HYBRID = "hybrid"              # Combination of both


class AgentType(str, Enum):
    """Types of agents in the system."""

    LLM = "llm"                    # AI/LLM agent
    SOFTWARE = "software"          # Traditional software agent
    HUMAN = "human"                # Human-in-the-loop


class IOFormat(str, Enum):
    """Input/Output format types."""

    JSON = "json"
    TEXT = "text"
    BINARY = "binary"
    MULTIMODAL = "multimodal"
    STREAMING = "streaming"


class DiscoveryTier(str, Enum):
    """Discovery mechanism tiers."""

    LOCAL = "local"      # mDNS local network discovery
    CLUSTER = "cluster"  # Gossip protocol for cluster
    GLOBAL = "global"    # DHT for global discovery


class TransportProtocol(str, Enum):
    """Transport protocols supported."""

    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    MQTT = "mqtt"
    AMQP = "amqp"
    WEBSOCKET = "websocket"


class NegotiationStatus(str, Enum):
    """Status of a negotiation."""

    INITIATED = "initiated"
    OFFERED = "offered"
    COUNTERED = "countered"
    ACCEPTED = "accepted"
    CONFIRMED = "confirmed"
    READY = "ready"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TrustLevel(str, Enum):
    """Trust level classifications."""

    UNTRUSTED = "untrusted"     # 0.0 - 0.2
    LOW = "low"                 # 0.2 - 0.4
    MEDIUM = "medium"           # 0.4 - 0.6
    HIGH = "high"               # 0.6 - 0.8
    VERIFIED = "verified"       # 0.8 - 1.0


class ValidationType(str, Enum):
    """Types of capability validation."""

    SCHEMA = "schema"           # JSON schema validation
    RUNTIME = "runtime"         # Runtime capability test
    ATTESTATION = "attestation" # Cryptographic proof
    SIMULATION = "simulation"   # Simulated test execution
