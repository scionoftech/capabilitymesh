"""Agent identity and addressing system."""

import hashlib
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from capabilitymesh.core.types import AgentType
from capabilitymesh.core.exceptions import InvalidDIDError


class AgentAddress(BaseModel):
    """Network address for an agent."""

    protocol: str  # "http", "grpc", "mqtt", etc.
    host: str
    port: int
    path: Optional[str] = None

    def to_uri(self) -> str:
        """Convert to URI string."""
        uri = f"{self.protocol}://{self.host}:{self.port}"
        if self.path:
            uri += self.path
        return uri

    @classmethod
    def from_uri(cls, uri: str) -> "AgentAddress":
        """Create AgentAddress from URI string.

        Args:
            uri: URI string like "http://localhost:8000/path"

        Returns:
            AgentAddress instance

        Raises:
            ValueError: If URI format is invalid
        """
        try:
            # Simple URI parsing
            if "://" not in uri:
                raise ValueError("Invalid URI: missing protocol")

            protocol, rest = uri.split("://", 1)

            # Extract path if present
            if "/" in rest:
                host_port, path = rest.split("/", 1)
                path = "/" + path
            else:
                host_port = rest
                path = None

            # Extract host and port
            if ":" in host_port:
                host, port_str = host_port.rsplit(":", 1)
                port = int(port_str)
            else:
                host = host_port
                # Default ports
                port = 443 if protocol == "https" else 80

            return cls(protocol=protocol, host=host, port=port, path=path)

        except Exception as e:
            raise ValueError(f"Invalid URI format: {uri}") from e


class AgentIdentity(BaseModel):
    """Unique identity for an agent in the ACDP network.

    Uses DID (Decentralized Identifier) compatible format: did:acdp:{hash}
    """

    # Core identity
    did: str  # Format: did:acdp:{hash}
    name: str
    agent_type: AgentType

    # Network addresses
    addresses: List[AgentAddress]
    primary_address: AgentAddress

    # Public key for verification
    public_key: str  # PEM format
    key_algorithm: str = "RSA-2048"

    # Metadata
    description: Optional[str] = None
    organization: Optional[str] = None
    homepage: Optional[str] = None

    # Discovery metadata
    discovery_enabled: bool = True
    advertise_capabilities: bool = True

    # Trust information
    reputation_score: float = 0.0  # 0.0 to 1.0
    verified: bool = False
    verification_authority: Optional[str] = None

    # Timestamps
    created_at: datetime
    last_seen: datetime

    @property
    def id(self) -> str:
        """Get the agent's unique ID (same as DID)."""
        return self.did

    @classmethod
    def create_simple(
        cls,
        name: str,
        agent_type: AgentType,
        description: Optional[str] = None,
    ) -> "AgentIdentity":
        """Create a simple AgentIdentity with auto-generated fields.

        This is a convenience method for v1.0 that auto-generates:
        - DID
        - Public key (placeholder)
        - Primary address (local)
        - Timestamps

        Args:
            name: Agent name
            agent_type: Agent type
            description: Optional description

        Returns:
            AgentIdentity with auto-generated fields
        """
        # Generate a simple public key placeholder
        import uuid
        key_id = uuid.uuid4().hex
        public_key = f"-----BEGIN PUBLIC KEY-----\n{key_id}\n-----END PUBLIC KEY-----"

        # Generate DID
        did = cls.generate_did(public_key)

        # Create local address
        primary_addr = AgentAddress(
            protocol="local",
            host="localhost",
            port=0,
            path=f"/{name.replace(' ', '-').lower()}",
        )

        # Create identity
        now = datetime.utcnow()
        return cls(
            did=did,
            name=name,
            agent_type=agent_type,
            addresses=[primary_addr],
            primary_address=primary_addr,
            public_key=public_key,
            description=description,
            created_at=now,
            last_seen=now,
        )

    @classmethod
    def generate_did(cls, public_key: str) -> str:
        """Generate DID from public key.

        Args:
            public_key: PEM-formatted public key

        Returns:
            DID string in format did:acdp:{hash}
        """
        key_hash = hashlib.sha256(public_key.encode()).hexdigest()[:32]
        return f"did:acdp:{key_hash}"

    @classmethod
    def validate_did(cls, did: str) -> bool:
        """Validate DID format.

        Args:
            did: DID string to validate

        Returns:
            True if valid, False otherwise
        """
        if not did.startswith("did:acdp:"):
            return False

        # Extract hash part
        try:
            _, method, identifier = did.split(":", 2)
            if method != "acdp":
                return False
            # Hash should be 32 hex characters
            if len(identifier) != 32 or not all(c in "0123456789abcdef" for c in identifier):
                return False
            return True
        except ValueError:
            return False

    def model_post_init(self, __context: object) -> None:
        """Validate after model initialization."""
        if not self.validate_did(self.did):
            raise InvalidDIDError(f"Invalid DID format: {self.did}")

        if self.primary_address not in self.addresses:
            raise ValueError("Primary address must be in addresses list")

        if not 0.0 <= self.reputation_score <= 1.0:
            raise ValueError("Reputation score must be between 0.0 and 1.0")

    def update_last_seen(self) -> None:
        """Update the last_seen timestamp to current time."""
        self.last_seen = datetime.now()

    def add_address(self, address: AgentAddress, set_as_primary: bool = False) -> None:
        """Add a new network address.

        Args:
            address: AgentAddress to add
            set_as_primary: Whether to set this as the primary address
        """
        if address not in self.addresses:
            self.addresses.append(address)

        if set_as_primary:
            self.primary_address = address

    def remove_address(self, address: AgentAddress) -> None:
        """Remove a network address.

        Args:
            address: AgentAddress to remove

        Raises:
            ValueError: If trying to remove the primary address or address not found
        """
        if address == self.primary_address:
            raise ValueError("Cannot remove primary address")

        if address in self.addresses:
            self.addresses.remove(address)
        else:
            raise ValueError("Address not found")
