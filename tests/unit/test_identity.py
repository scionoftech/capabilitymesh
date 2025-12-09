"""Unit tests for agent identity and addressing."""

import pytest
from datetime import datetime

from capabilitymesh.core.identity import AgentIdentity, AgentAddress
from capabilitymesh.core.types import AgentType
from capabilitymesh.core.exceptions import InvalidDIDError


class TestAgentAddress:
    """Tests for AgentAddress class."""

    def test_create_agent_address(self):
        """Test creating an agent address."""
        address = AgentAddress(protocol="http", host="localhost", port=8000, path="/api")

        assert address.protocol == "http"
        assert address.host == "localhost"
        assert address.port == 8000
        assert address.path == "/api"

    def test_to_uri(self):
        """Test converting address to URI."""
        address = AgentAddress(protocol="https", host="example.com", port=443, path="/v1/api")
        uri = address.to_uri()

        assert uri == "https://example.com:443/v1/api"

    def test_to_uri_without_path(self):
        """Test URI conversion without path."""
        address = AgentAddress(protocol="http", host="localhost", port=8000)
        uri = address.to_uri()

        assert uri == "http://localhost:8000"

    def test_from_uri(self):
        """Test creating address from URI."""
        uri = "http://localhost:8000/api/v1"
        address = AgentAddress.from_uri(uri)

        assert address.protocol == "http"
        assert address.host == "localhost"
        assert address.port == 8000
        assert address.path == "/api/v1"

    def test_from_uri_without_path(self):
        """Test creating address from URI without path."""
        uri = "grpc://example.com:9000"
        address = AgentAddress.from_uri(uri)

        assert address.protocol == "grpc"
        assert address.host == "example.com"
        assert address.port == 9000
        assert address.path is None

    def test_from_uri_default_port(self):
        """Test URI parsing with default port."""
        uri = "http://example.com"
        address = AgentAddress.from_uri(uri)

        assert address.port == 80

        uri_https = "https://example.com"
        address_https = AgentAddress.from_uri(uri_https)

        assert address_https.port == 443

    def test_from_uri_invalid(self):
        """Test creating address from invalid URI."""
        with pytest.raises(ValueError):
            AgentAddress.from_uri("invalid-uri")


class TestAgentIdentity:
    """Tests for AgentIdentity class."""

    @pytest.fixture
    def sample_public_key(self):
        """Sample public key for testing."""
        return """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1234567890
-----END PUBLIC KEY-----"""

    @pytest.fixture
    def sample_address(self):
        """Sample agent address."""
        return AgentAddress(protocol="http", host="localhost", port=8000)

    def test_generate_did(self, sample_public_key):
        """Test DID generation from public key."""
        did = AgentIdentity.generate_did(sample_public_key)

        assert did.startswith("did:acdp:")
        assert len(did) == 41  # "did:acdp:" (9) + 32 char hash

    def test_generate_did_deterministic(self, sample_public_key):
        """Test that DID generation is deterministic."""
        did1 = AgentIdentity.generate_did(sample_public_key)
        did2 = AgentIdentity.generate_did(sample_public_key)

        assert did1 == did2

    def test_validate_did_valid(self):
        """Test DID validation with valid DIDs."""
        valid_did = "did:acdp:0123456789abcdef0123456789abcdef"
        assert AgentIdentity.validate_did(valid_did) is True

    def test_validate_did_invalid(self):
        """Test DID validation with invalid DIDs."""
        invalid_dids = [
            "invalid:acdp:123",
            "did:other:0123456789abcdef0123456789abcdef",
            "did:acdp:short",
            "did:acdp:GHIJKLMNOPQRSTUVWXYZ12345678901",  # Invalid hex
            "did:acdp",
        ]

        for invalid_did in invalid_dids:
            assert AgentIdentity.validate_did(invalid_did) is False

    def test_create_agent_identity(self, sample_public_key, sample_address):
        """Test creating an agent identity."""
        did = AgentIdentity.generate_did(sample_public_key)
        now = datetime.now()

        identity = AgentIdentity(
            did=did,
            name="TestAgent",
            agent_type=AgentType.LLM,
            addresses=[sample_address],
            primary_address=sample_address,
            public_key=sample_public_key,
            created_at=now,
            last_seen=now,
        )

        assert identity.did == did
        assert identity.name == "TestAgent"
        assert identity.agent_type == AgentType.LLM
        assert len(identity.addresses) == 1
        assert identity.primary_address == sample_address

    def test_create_agent_identity_invalid_did(self, sample_public_key, sample_address):
        """Test that creating an identity with invalid DID raises error."""
        now = datetime.now()

        with pytest.raises(InvalidDIDError):
            AgentIdentity(
                did="invalid:did:format",
                name="TestAgent",
                agent_type=AgentType.LLM,
                addresses=[sample_address],
                primary_address=sample_address,
                public_key=sample_public_key,
                created_at=now,
                last_seen=now,
            )

    def test_primary_address_not_in_list(self, sample_public_key, sample_address):
        """Test that primary address must be in addresses list."""
        did = AgentIdentity.generate_did(sample_public_key)
        now = datetime.now()

        other_address = AgentAddress(protocol="http", host="other.com", port=9000)

        with pytest.raises(ValueError, match="Primary address must be in addresses list"):
            AgentIdentity(
                did=did,
                name="TestAgent",
                agent_type=AgentType.LLM,
                addresses=[sample_address],
                primary_address=other_address,
                public_key=sample_public_key,
                created_at=now,
                last_seen=now,
            )

    def test_invalid_reputation_score(self, sample_public_key, sample_address):
        """Test that reputation score must be between 0 and 1."""
        did = AgentIdentity.generate_did(sample_public_key)
        now = datetime.now()

        with pytest.raises(ValueError, match="Reputation score must be between 0.0 and 1.0"):
            AgentIdentity(
                did=did,
                name="TestAgent",
                agent_type=AgentType.LLM,
                addresses=[sample_address],
                primary_address=sample_address,
                public_key=sample_public_key,
                reputation_score=1.5,
                created_at=now,
                last_seen=now,
            )

    def test_update_last_seen(self, sample_public_key, sample_address):
        """Test updating last_seen timestamp."""
        import time
        did = AgentIdentity.generate_did(sample_public_key)
        now = datetime.now()

        identity = AgentIdentity(
            did=did,
            name="TestAgent",
            agent_type=AgentType.LLM,
            addresses=[sample_address],
            primary_address=sample_address,
            public_key=sample_public_key,
            created_at=now,
            last_seen=now,
        )

        original_last_seen = identity.last_seen
        time.sleep(0.01)  # Small delay to ensure timestamp changes
        identity.update_last_seen()

        assert identity.last_seen > original_last_seen

    def test_add_address(self, sample_public_key, sample_address):
        """Test adding a new address."""
        did = AgentIdentity.generate_did(sample_public_key)
        now = datetime.now()

        identity = AgentIdentity(
            did=did,
            name="TestAgent",
            agent_type=AgentType.LLM,
            addresses=[sample_address],
            primary_address=sample_address,
            public_key=sample_public_key,
            created_at=now,
            last_seen=now,
        )

        new_address = AgentAddress(protocol="grpc", host="grpc.example.com", port=9000)
        identity.add_address(new_address)

        assert len(identity.addresses) == 2
        assert new_address in identity.addresses

    def test_add_address_as_primary(self, sample_public_key, sample_address):
        """Test adding a new address and setting it as primary."""
        did = AgentIdentity.generate_did(sample_public_key)
        now = datetime.now()

        identity = AgentIdentity(
            did=did,
            name="TestAgent",
            agent_type=AgentType.LLM,
            addresses=[sample_address],
            primary_address=sample_address,
            public_key=sample_public_key,
            created_at=now,
            last_seen=now,
        )

        new_address = AgentAddress(protocol="grpc", host="grpc.example.com", port=9000)
        identity.add_address(new_address, set_as_primary=True)

        assert identity.primary_address == new_address

    def test_remove_address(self, sample_public_key, sample_address):
        """Test removing an address."""
        did = AgentIdentity.generate_did(sample_public_key)
        now = datetime.now()

        second_address = AgentAddress(protocol="grpc", host="grpc.example.com", port=9000)

        identity = AgentIdentity(
            did=did,
            name="TestAgent",
            agent_type=AgentType.LLM,
            addresses=[sample_address, second_address],
            primary_address=sample_address,
            public_key=sample_public_key,
            created_at=now,
            last_seen=now,
        )

        identity.remove_address(second_address)

        assert len(identity.addresses) == 1
        assert second_address not in identity.addresses

    def test_cannot_remove_primary_address(self, sample_public_key, sample_address):
        """Test that removing primary address raises error."""
        did = AgentIdentity.generate_did(sample_public_key)
        now = datetime.now()

        identity = AgentIdentity(
            did=did,
            name="TestAgent",
            agent_type=AgentType.LLM,
            addresses=[sample_address],
            primary_address=sample_address,
            public_key=sample_public_key,
            created_at=now,
            last_seen=now,
        )

        with pytest.raises(ValueError, match="Cannot remove primary address"):
            identity.remove_address(sample_address)

    def test_remove_nonexistent_address(self, sample_public_key, sample_address):
        """Test removing an address that doesn't exist."""
        did = AgentIdentity.generate_did(sample_public_key)
        now = datetime.now()

        identity = AgentIdentity(
            did=did,
            name="TestAgent",
            agent_type=AgentType.LLM,
            addresses=[sample_address],
            primary_address=sample_address,
            public_key=sample_public_key,
            created_at=now,
            last_seen=now,
        )

        nonexistent_address = AgentAddress(protocol="http", host="nowhere.com", port=1234)

        with pytest.raises(ValueError, match="Address not found"):
            identity.remove_address(nonexistent_address)
